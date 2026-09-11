"""Chinese figure typography shared by the third-question plotting entry points."""

from __future__ import annotations

import warnings



def configure_chinese(plt):
    """Select an installed Chinese font and preserve the existing figure geometry."""
    from matplotlib import font_manager

    installed = {item.name for item in font_manager.fontManager.ttflist}
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC")
    font = next((name for name in candidates if name in installed), None)
    if font is None:
        raise RuntimeError("需要中文字体：Microsoft YaHei、SimHei 或 Noto Sans CJK SC。")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "path",
    })
    warnings.filterwarnings("error", message="Glyph .* missing from font.*")
    return font
