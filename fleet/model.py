"""Pointer actor-critic for fleet dispatch.

Vehicles are coordinated with a transformer (guests → vehicles). Actions are a
masked pointer over padded request slots plus STAY/IDLE — not a softmax over
intersections. Optional mean-aggregation GNN encodes the street graph into
fixed-width node embeddings used for vehicle/request locations.
"""

import torch
import torch.nn as nn

import fleet.config as config