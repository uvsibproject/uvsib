"""ASE calculator factory used by relax / face_build / adsorbates / mh.

Shipped as a sidecar alongside each ``aiida.py`` script (see the CalcJob
``prepare_for_submission`` of mace / mattersim / upet / uma / sevennet /
minimahopping). Each script does ``from _calculators import make_calculator``
and the workchain controls the backend + UMA task head via ``--ML_model`` and
``--task_name``.
"""

import os

_UMA_TASKS = ("omat", "oc20", "oc22", "omol", "odac", "omc")
_MACE_TASKS = ("omat_pbe", "omol", "spice_wB97M", "rgd1_b3lyp", "oc20_usemppbe", "matpes_r2scan")

# SevenNet (https://github.com/MDIL-SNU/SevenNet) pretrained keywords understood
# by ``sevenn.calculator.SevenNetCalculator``. A local checkpoint path passed via
# ``--model_path`` overrides these; the list is informational only (SevenNet
# accepts other keywords / released checkpoints too, so it is not enforced).
_SEVENNET_MODELS = ("7net-0", "7net-l3i5", "7net-mf-ompa", "7net-omat", "7net-omni", "7net-nano-5.5")
# ``modal`` values for the multi-fidelity SevenNet checkpoints (7net-mf-ompa /
# 7net-omni); ignored by the single-fidelity ones. Passed through from
# ``--task_name`` when set to a real value.
_SEVENNET_MODALS = ("mpa", "omat24")

def make_calculator(ml_model, *, model=None, model_path=None, device="cuda",
                    task_name=None):
    """Build an ASE calculator from the workflow-supplied backend identifier.

    Parameters
    ----------
    ml_model : str
        Backend tag passed in via ``--ML_model``. Recognised tokens (substring
        match, matching the workchain dispatch):
        ``MACE``, ``uPET``, ``MatterSim``, ``UMA``, ``SevenNet``.
    model : str | None
        HuggingFace-style model name / pretrained keyword (``--model``); used by
        uPET, UMA and SevenNet (e.g. ``7net-0``, ``7net-mf-ompa``).
    model_path : str | None
        Local path to a checkpoint (``--model_path``); used by MACE, MatterSim
        and SevenNet. For SevenNet an existing path takes precedence over the
        ``model`` keyword.
    device : str
        Torch device string.
    task_name : str | None
        UMA task head (must be one of ``{_UMA_TASKS}``), or the SevenNet
        ``modal`` selector for multi-fidelity checkpoints (e.g. ``mpa``,
        ``omat24``). Ignored for the other backends and when unset / ``"None"``.
    """
    if "MACE" in ml_model:
        if task_name not in _MACE_TASKS:
            raise ValueError(f"Unknown MACE task_name '{task_name}'. Expected one of: {', '.join(_MACE_TASKS)}.")
        from mace.calculators import mace_mp
        return mace_mp(model=model_path, default_dtype="float64", device=device, head=task_name)

    if "uPET" in ml_model:
        from upet.calculator import UPETCalculator
        return UPETCalculator(model=model, device=device)

    if "MatterSim" in ml_model:
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(load_path=model_path, device=device)

    if "UMA" in ml_model:
        if task_name not in _UMA_TASKS:
            raise ValueError(f"Unknown UMA task_name '{task_name}'. Expected one of: {', '.join(_UMA_TASKS)}.")
        from fairchem.core import pretrained_mlip
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit(model, device=device)
        return FAIRChemCalculator(predictor, task_name=task_name)

    if "SevenNet" in ml_model:
        try:
            from sevenn.calculator import SevenNetCalculator
        except ImportError:  # sevenn < 0.10 shipped it under a different module
            from sevenn.sevennet_calculator import SevenNetCalculator
        # Prefer an existing local checkpoint; otherwise fall back to the
        # pretrained keyword (``--model``), defaulting to the SevenNet-0
        # universal potential.
        if model_path and os.path.exists(model_path):
            checkpoint = model_path
        else:
            checkpoint = model or "7net-0"
        kwargs = {"device": device}
        # ``get_cmdline`` always forwards ``--task_name``; treat an unset value
        # (Python ``None`` stringified, empty) as "no modal".
        if task_name and str(task_name).lower() not in ("none", ""):
            kwargs["modal"] = task_name
        return SevenNetCalculator(checkpoint, **kwargs)

    raise ValueError(
        f"Unknown ML_model '{ml_model}'. Expected one of: "
        "MACE, uPET, UMA, MatterSim, SevenNet.")
