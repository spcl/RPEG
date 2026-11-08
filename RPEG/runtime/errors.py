# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Pierre Pang

class RuntimeConfigError(ValueError):
    """
    Raised when a runtime YAML or optimized plan is invalid.
    """


class RuntimeDependencyError(RuntimeError):
    """
    Raised when an optional runtime dependency is required but unavailable.
    """


class RuntimeExecutionError(RuntimeError):
    """
    Raised when runtime execution fails after validation.
    """
