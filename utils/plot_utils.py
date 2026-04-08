import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager


def set_chinese_font() -> None:
    candidates = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Source Han Sans CN",
        "Source Han Sans SC",
        "STHeiti",
    ]
    available = {f.name for f in fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
