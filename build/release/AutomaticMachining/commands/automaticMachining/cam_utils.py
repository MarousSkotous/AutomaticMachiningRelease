"""Small shared helpers for building CAM operations."""

from __future__ import annotations

import adsk.cam


def set_expression(parameters: adsk.cam.CAMParameters, name: str, expression: str) -> None:
    parameter = parameters.itemByName(name)
    if parameter:
        parameter.expression = expression


def set_safe_heights(parameters: adsk.cam.CAMParameters) -> None:
    """Sets a fixed, valid rapid/feed height order above the exact-size stock."""
    set_expression(parameters, 'clearanceHeight_mode', "'from retract height'")
    set_expression(parameters, 'clearanceHeight_offset', '5 mm')
    set_expression(parameters, 'retractHeight_mode', "'from stock top'")
    set_expression(parameters, 'retractHeight_offset', '5 mm')
    set_expression(parameters, 'feedHeight_mode', "'from stock top'")
    set_expression(parameters, 'feedHeight_offset', '2 mm')


def set_zero_stock_to_leave(parameters: adsk.cam.CAMParameters) -> None:
    """Enables both stock fields and sets their actual Fusion values to zero."""
    set_expression(parameters, 'useStockToLeave', 'true')
    set_expression(parameters, 'stockToLeave', '0 mm')
    set_expression(parameters, 'verticalStockToLeave', '0 mm')
