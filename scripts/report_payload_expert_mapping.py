"""Measure fixed load meanings of incoming expert traffic; no router fitting."""

import argparse
import json
from pathlib import Path

import torch

from intact_tracking.payload_prototype_moe import PrototypeGate


def routing_mapping(ids, weights, actual_masses, grid, *, anchor_ids=None):
    ids, weights, actual_masses, grid = [x.cpu() for x in (ids, weights, actual_masses, grid)]
    limits = grid.max(0).values
    selected = grid[ids]
    top = selected[:,0]
    weighted = (weights[...,None]*selected).sum(1)
    error = (top-actual_masses).abs()
    normalized_distance = torch.linalg.vector_norm((grid[None]-actual_masses[:,None])/limits,dim=-1)
    nearest = normalized_distance.argmin(-1)
    coarse_actual=actual_masses>=limits/2
    coarse_top=top>=limits/2
    coarse_equal=coarse_top==coarse_actual
    expected_error = (weights[...,None]*(selected-actual_masses[:,None]).abs()).sum(1)
    occupancy = torch.bincount(ids[:,0],minlength=256)
    traffic = torch.zeros(256).scatter_add_(0,ids.flatten(),weights.flatten())
    incoming_sum = torch.zeros(256,4).index_add_(0,ids.flatten(),(weights[...,None]*actual_masses[:,None]).flatten(0,1))
    incoming_sq = torch.zeros(256,4).index_add_(0,ids.flatten(),(weights[...,None]*actual_masses[:,None].square()).flatten(0,1))
    incoming_mean = incoming_sum/traffic[:,None].clamp_min(1e-8)
    incoming_std = (incoming_sq/traffic[:,None].clamp_min(1e-8)-incoming_mean.square()).clamp_min(0).sqrt()
    row = {'samples':len(ids),'top1_used_experts':int((occupancy>0).sum()),
           'top5_used_experts':int((traffic>0).sum()),'top1_mass_mae_kg':error.mean(0).tolist(),
           'weighted_mass_mae_kg':(weighted-actual_masses).abs().mean(0).tolist(),
           'mean_selected_expert_absolute_mass_error_kg':expected_error.mean(0).tolist(),
           'nearest_physical_grid_top1':float((ids[:,0]==nearest).float().mean()),
           'nearest_physical_grid_top5':float((ids==nearest[:,None]).any(1).float().mean()),
           'light_heavy_per_limb_accuracy':coarse_equal.float().mean(0).tolist(),
           'light_heavy_all_four_accuracy':float(coarse_equal.all(1).float().mean()),
           'both_hand_light_heavy_accuracy':float(coarse_equal[:,:2].all(1).float().mean()),
           'top1_normalized_load_distance':float(normalized_distance.gather(1,ids[:,:1]).mean()),
           'nearest_possible_grid_normalized_load_distance':float(normalized_distance.min(1).values.mean()),
           'weighted_mean_normalized_expert_distance':float((weights*normalized_distance.gather(1,ids)).sum(1).mean()),
           'incoming_expert_traffic_weight':traffic.tolist(),'incoming_expert_mean_masses_kg':incoming_mean.tolist(),
           'incoming_expert_mass_std_kg':incoming_std.tolist()}
    if anchor_ids is not None:
        y=anchor_ids.cpu().long()
        if not torch.all((actual_masses-grid[y]).abs()<1e-5):
            raise ValueError('Anchor load identities do not match the actual physics')
        correct = ids==y[:,None]
        confusion = torch.bincount(y*256+ids[:,0],minlength=256*256).reshape(256,256)
        soft = torch.zeros(256*256).scatter_add_(0,(y[:,None]*256+ids).flatten(),weights.flatten()).reshape(256,256)
        counts=confusion.sum(1)
        valid=counts>0
        fractions=confusion/counts[:,None].clamp_min(1)
        soft_fractions=soft/counts[:,None].clamp_min(1)
        dominant=confusion.argmax(1)
        correct_classes=(dominant==torch.arange(256))&valid
        row.update(exact_grid_top1=float(correct[:,0].float().mean()),
            exact_grid_top5=float(correct.any(1).float().mean()),
            true_expert_mean_weight=float((correct*weights).sum(1).mean()),
            per_limb_top1_exact_level=((top-actual_masses).abs()<1e-5).float().mean(0).tolist(),
            classes_observed=int(valid.sum()),classes_correct_modal_expert=int(correct_classes.sum()),
            classes_correct_weighted_dominant_expert=int(((soft.argmax(1)==torch.arange(256))&valid).sum()),
            incoming_top1_purity=float(confusion.max(0).values.sum()/confusion.sum()),
            incoming_soft_correct_semantic_fraction=float(soft.diag().sum()/soft.sum()),
            top1_confusion_counts=confusion.tolist(),soft_confusion_weights=soft.tolist(),
            per_class=[{'id':i,'masses_kg':grid[i].tolist(),'samples':int(counts[i]),
                'correct_top1_fraction':float(fractions[i,i]),'true_expert_mean_weight':float(soft_fractions[i,i]),
                'modal_expert':int(dominant[i]),'modal_expert_mass_kg':grid[dominant[i]].tolist(),
                'top3_experts':confusion[i].topk(3).indices.tolist(),
                'top3_fractions':fractions[i].topk(3).values.tolist()} for i in range(256)])
    return row


def compact(row):
    return {k:v for k,v in row.items() if k not in ('top1_confusion_counts','soft_confusion_weights','per_class')
            and not k.startswith('incoming_expert_')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path(args.root)
    proto=torch.load(root/'calibration/prototypes.pt',map_location='cpu',weights_only=False)
    samples=torch.load(root/'calibration/calibration_samples.pt',map_location='cpu',weights_only=False)
    mask=((samples['worlds']//256)%4==3)&samples['full'].flatten()
    z=samples['latent'].reshape(-1,64)[mask]
    y=samples['ids'][mask]
    ids,w=PrototypeGate(proto['centers'],proto['temperature'])(z)
    result={'protocol':'held_out_whole_worlds_frozen_tracker_calibration',
            'context_sha256':proto['metadata']['context_sha256'],'temperature':proto['temperature'],
            'mapping':routing_mapping(ids,w,proto['loads_kg'][y],proto['loads_kg'],anchor_ids=y)}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(compact(result['mapping'])),flush=True)


if __name__=='__main__':main()
