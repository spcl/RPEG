# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

from typing import Optional

from core.types import Container, EdgeKind, Model, OperatorKind, Role, TemplateEdge, TemplateVertex
from ir.rpeg import RPEG
from model.models import ModelSuite
from model.workload import Workload


def build_PPO_RPEG(suite: ModelSuite, workload: Workload, tau: Optional[int] = None, verbose: bool = False) -> RPEG:
    """
    Construct the PPO (Proximal Policy Optimization) RPEG.

    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param tau: Staleness budget. Defaults to None.
    :type tau: Optional[int]
    :param verbose: Verbose flag for additional output. Defaults to False.
    :type verbose: bool
    :return: RLM Pipeline Execution Graph.
    :rtype: RPEG
    """
    B, K, S, T = workload.B, workload.K, workload.S, workload.T  
    trainable = {Model.ACTOR, Model.CRITIC}
    
    # Transient containers (within-step data)
    X = Container(name="prompts", count=B * S, elem_size_bytes=2)
    Y = Container(name="completions", count=B * K * (S + T), elem_size_bytes=2)
    log_pi = Container(name="log_pi", count=B * K * T, elem_size_bytes=2)
    log_ref = Container(name="log_ref", count=B * K * T, elem_size_bytes=2)
    r = Container(name="rewards", count=B * K, elem_size_bytes=2)
    v = Container(name="values", count=B * K * T, elem_size_bytes=2)

    # Weight containers (cross-step via publication edges)
    W_pi = Container(name="W_pi", count=suite.n_params(Model.ACTOR), elem_size_bytes=2)
    W_V = Container(name="W_V", count=suite.n_params(Model.CRITIC), elem_size_bytes=2)
    W_ref = Container(name="W_ref", count=suite.n_params(Model.REFERENCE), elem_size_bytes=2)
    W_R = Container(name="W_R", count=suite.n_params(Model.REWARD), elem_size_bytes=2)
    
    containers = (Y, log_pi, log_ref, r, v, W_pi, W_V, W_ref, W_R)
    in_containers = (X, W_ref, W_R)
    out_containers = () 

    # Operator kinds
    o1 = OperatorKind(role=Role.GENERATION, model=Model.ACTOR, read=(W_pi, X), write=(Y, log_pi))
    o2 = OperatorKind(role=Role.INFERENCE, model=Model.REFERENCE, read=(W_ref, Y), write=(log_ref, ))
    o3 = OperatorKind(role=Role.INFERENCE, model=Model.REWARD, read=(W_R, Y), write=(r, ))
    o4 = OperatorKind(role=Role.INFERENCE, model=Model.CRITIC, read=(W_V, Y), write=(v, ))
    o5 = OperatorKind(role=Role.TRAINING, model=Model.ACTOR, read=(W_pi, Y, log_pi, log_ref, r, v), write=(W_pi, ))
    o6 = OperatorKind(role=Role.TRAINING, model=Model.CRITIC, read=(W_V, Y, log_ref, r, v), write=(W_V, ))

    # Vertices
    g_pi = TemplateVertex(id="g_pi", kind=o1)
    i_ref = TemplateVertex(id="i_ref", kind=o2)
    i_R = TemplateVertex(id="i_R", kind=o3)
    i_V = TemplateVertex(id="i_V", kind=o4)
    t_pi = TemplateVertex(id="t_pi", kind=o5)
    t_V = TemplateVertex(id="t_V", kind=o6)

    vertices = (g_pi, i_ref, i_R, i_V, t_pi, t_V) 

    # Edges
    # generation -> inference
    e1 = TemplateEdge(src=g_pi, dst=i_ref, container=Y, kind=EdgeKind.DATA)
    e2 = TemplateEdge(src=g_pi, dst=i_R, container=Y, kind=EdgeKind.DATA)
    e3 = TemplateEdge(src=g_pi, dst=i_V, container=Y, kind=EdgeKind.DATA)

    # inference -> training
    # No direct g_pi -> t_pi/t_V edges: training reads Y and log_pi from memory,
    # decoupling its scheduling from generation.
    e4 = TemplateEdge(src=i_ref, dst=t_pi, container=log_ref, kind=EdgeKind.DATA)
    e5 = TemplateEdge(src=i_ref, dst=t_V, container=log_ref, kind=EdgeKind.DATA)
    e6 = TemplateEdge(src=i_R, dst=t_pi, container=r, kind=EdgeKind.DATA)
    e7 = TemplateEdge(src=i_V, dst=t_pi, container=v, kind=EdgeKind.DATA)
    e8 = TemplateEdge(src=i_R, dst=t_V, container=r, kind=EdgeKind.DATA)
    e9 = TemplateEdge(src=i_V, dst=t_V, container=v, kind=EdgeKind.DATA)

    # training -> generation | inference
    e10 = TemplateEdge(src=t_pi, dst=g_pi, container=W_pi, kind=EdgeKind.PUB)
    e11 = TemplateEdge(src=t_V, dst=i_V, container=W_V, kind=EdgeKind.PUB)

    data_edges = (e1, e2, e3, e4, e5, e6, e7, e8, e9)
    pub_edges = (e10, e11) 

    # Staleness
    # Default: synchronous (τ=0). With τ>0 the actor inference could run on stale weights while the
    # previous training step finishes.
    if tau is None: 
        tau = {Model.ACTOR: 0, Model.CRITIC: 0} 

    # PPO RPEG
    ppo_rpeg = RPEG(vertices, data_edges, pub_edges, containers, in_containers, out_containers,tau, trainable)
    
    ppo_rpeg.well_formed(verbose)
    
    if verbose: 
        print("RPEG instantiated successfully for PPO.")
        print(f"    Template Nodes: {len(ppo_rpeg.vertices)}.")
        print(f"    Data edges: {len(ppo_rpeg.data_edges)}.")
        print(f"    Publication edges: {len(ppo_rpeg.pub_edges)}.")
        print(f"    Container: {len(ppo_rpeg.containers)}.")
        print(f"    IN_Container: {len(ppo_rpeg.in_containers)}.")
        print(f"    OUT_Container: {len(ppo_rpeg.out_containers)}.")

    return ppo_rpeg
