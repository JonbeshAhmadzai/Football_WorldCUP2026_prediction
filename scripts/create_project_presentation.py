from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import uno
from PIL import Image, ImageDraw, ImageFont
from com.sun.star.awt import Point, Size
from com.sun.star.beans import PropertyValue


BASE_DIR = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = BASE_DIR / "presentations"
ASSET_DIR = PRESENTATION_DIR / "assets"
OUTPUT_PATH = PRESENTATION_DIR / "football_match_prediction_project_5_slides.pptx"
PREVIEW_PDF_PATH = PRESENTATION_DIR / "football_match_prediction_project_5_slides.pdf"

SLIDE_W = 28000
SLIDE_H = 15750

NAVY = 0x101828
INK = 0x1D2939
MUTED = 0x667085
PAPER = 0xF8FAFC
WHITE = 0xFFFFFF
BLUE = 0x2563EB
CYAN = 0x06B6D4
GREEN = 0x16A34A
AMBER = 0xF59E0B
RED = 0xDC2626
PURPLE = 0x7C3AED
LINE = 0xD0D5DD

IMG_W = 1600
IMG_H = 900
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def file_url(path: Path) -> str:
    return uno.systemPathToFileUrl(str(path.resolve()))


def latest_json(pattern: str) -> dict:
    paths = sorted(BASE_DIR.glob(pattern), reverse=True)
    if not paths:
        return {}
    return json.loads(paths[0].read_text())


def load_project_numbers():
    results_path = BASE_DIR / "data" / "processed" / "worldcup_2026_live_results.csv"
    augmented_sim_path = (
        BASE_DIR
        / "src"
        / "modeling"
        / "simulation_runs"
        / "simulation_results_augmented_20260630_095848.csv"
    )
    live_sim_path = (
        BASE_DIR
        / "src"
        / "modeling"
        / "simulation_runs"
        / "simulation_results_live_only_20260630_095849.csv"
    )

    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    augmented_sim = pd.read_csv(augmented_sim_path) if augmented_sim_path.exists() else pd.DataFrame()
    live_sim = pd.read_csv(live_sim_path) if live_sim_path.exists() else pd.DataFrame()
    augmented_metrics = latest_json("src/modeling/artifacts/augmented_*/metrics.json")
    live_metrics = latest_json("src/modeling/artifacts/live_*/metrics.json")
    score_metrics = latest_json("src/modeling/score_artifacts/exact_score_augmented_*/metrics.json")

    return {
        "results": results,
        "augmented_sim": augmented_sim,
        "live_sim": live_sim,
        "augmented_metrics": augmented_metrics,
        "live_metrics": live_metrics,
        "score_metrics": score_metrics,
    }


def save_charts(numbers):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#D0D5DD",
            "axes.labelcolor": "#475467",
            "xtick.color": "#475467",
            "ytick.color": "#344054",
        }
    )

    results = numbers["results"]
    if not results.empty:
        counts = results["status"].value_counts()
        finished = int(results["is_finished"].sum())
        scheduled = int(results["is_scheduled"].sum())
        live = int(results["is_live"].sum())
    else:
        counts = pd.Series(dtype=int)
        finished = scheduled = live = 0

    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=180)
    bars = ax.bar(
        ["Finished", "Scheduled", "Live"],
        [finished, scheduled, live],
        color=["#16A34A", "#2563EB", "#F59E0B"],
        width=0.6,
    )
    ax.set_title("Latest ESPN Match Snapshot", loc="left", fontsize=15, weight="bold")
    ax.set_ylabel("Matches")
    ax.grid(axis="y", color="#EAECF0", linewidth=0.8)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color="#1D2939",
        )
    fig.tight_layout()
    snapshot_path = ASSET_DIR / "match_snapshot.png"
    fig.savefig(snapshot_path, transparent=True)
    plt.close(fig)

    sim = numbers["augmented_sim"]
    top_path = ASSET_DIR / "top_winner_probabilities.png"
    if not sim.empty:
        top = sim.sort_values("win_pct", ascending=False).head(7).sort_values("win_pct")
        fig, ax = plt.subplots(figsize=(7, 4.1), dpi=180)
        ax.barh(top["team"], top["win_pct"], color="#2563EB")
        ax.set_title("Augmented Model: Tournament Win Probability", loc="left", fontsize=15, weight="bold")
        ax.set_xlabel("Probability (%)")
        ax.grid(axis="x", color="#EAECF0", linewidth=0.8)
        for value, team in zip(top["win_pct"], top["team"]):
            ax.text(value + 0.4, team, f"{value:.1f}%", va="center", fontsize=10, color="#1D2939")
        fig.tight_layout()
        fig.savefig(top_path, transparent=True)
        plt.close(fig)

    metric_path = ASSET_DIR / "model_metrics.png"
    augmented_accuracy = numbers["augmented_metrics"].get("eval_accuracy", 0) * 100
    live_accuracy = numbers["live_metrics"].get("eval_accuracy", 0) * 100
    score_eval = numbers["score_metrics"].get("eval_metrics", {})
    top3_score = score_eval.get("top3_scoreline_accuracy", 0) * 100
    exact_score = score_eval.get("rounded_exact_score_accuracy", 0) * 100
    labels = ["Augmented\nresult", "Live result\nonly", "Scoreline\ntop 3", "Exact\nrounded"]
    values = [augmented_accuracy, live_accuracy, top3_score, exact_score]
    colors = ["#2563EB", "#06B6D4", "#7C3AED", "#F59E0B"]
    fig, ax = plt.subplots(figsize=(7, 3.9), dpi=180)
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title("Model Evaluation Highlights", loc="left", fontsize=15, weight="bold")
    ax.set_ylabel("Accuracy / hit rate (%)")
    ax.set_ylim(0, max(values + [60]) + 10)
    ax.grid(axis="y", color="#EAECF0", linewidth=0.8)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
            color="#1D2939",
        )
    fig.tight_layout()
    fig.savefig(metric_path, transparent=True)
    plt.close(fig)

    return {
        "snapshot": snapshot_path,
        "top": top_path,
        "metrics": metric_path,
        "status_counts": counts,
    }


def rgb(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def wrap_lines(draw: ImageDraw.ImageDraw, value: str, text_font, max_width: int) -> list[str]:
    words = value.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=text_font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    color: tuple[int, int, int],
    bold: bool = False,
    max_width: int | None = None,
    line_gap: int = 8,
):
    text_font = font(size, bold)
    x, y = xy
    if not max_width:
        draw.text((x, y), value, font=text_font, fill=color)
        return
    for line in wrap_lines(draw, value, text_font, max_width):
        draw.text((x, y), line, font=text_font, fill=color)
        y += size + line_gap


def rounded(draw, box, fill, outline=None, radius=14, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def paste_image(canvas: Image.Image, path: Path, box: tuple[int, int, int, int]):
    image_obj = Image.open(path).convert("RGBA")
    x1, y1, x2, y2 = box
    max_w = x2 - x1
    max_h = y2 - y1
    image_obj.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = x1 + (max_w - image_obj.width) // 2
    y = y1 + (max_h - image_obj.height) // 2
    canvas.alpha_composite(image_obj, (x, y))


def card(draw, box, title, value, accent, title_size=18, value_size=36):
    rounded(draw, box, rgb(WHITE), rgb(LINE), radius=10, width=2)
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x1 + 18, y2), fill=accent)
    draw_text(draw, (x1 + 42, y1 + 26), title, title_size, rgb(MUTED), False, x2 - x1 - 80)
    draw_text(draw, (x1 + 42, y1 + 76), value, value_size, rgb(INK), True, x2 - x1 - 80)


def bullet(draw, x, y, value, color=rgb(INK), max_width=580, size=24):
    draw.rounded_rectangle((x, y + 8, x + 14, y + 22), radius=4, fill=rgb(BLUE))
    draw_text(draw, (x + 34, y), value, size, color, False, max_width, line_gap=6)


def draw_header(draw, kicker, title, dark=False):
    kicker_color = rgb(CYAN if dark else BLUE)
    title_color = rgb(WHITE if dark else INK)
    draw_text(draw, (66, 48), kicker.upper(), 22, kicker_color, True)
    draw_text(draw, (66, 92), title, 54, title_color, True, 1100, line_gap=8)


def decorate(canvas: Image.Image, dark=False):
    draw = ImageDraw.Draw(canvas)
    bg = rgb(NAVY if dark else PAPER)
    draw.rectangle((0, 0, IMG_W, IMG_H), fill=bg)
    side = (24, 36, 91) if dark else (224, 242, 254)
    draw.rectangle((1260, 0, IMG_W, IMG_H), fill=side)
    draw.rectangle((1304, 72, 1488, 256), fill=rgb(CYAN))
    draw.rectangle((1354, 134, 1500, 280), fill=rgb(BLUE if not dark else PURPLE))
    draw.rectangle((0, 0, 18, IMG_H), fill=rgb(CYAN))


def render_slide_images(numbers, assets) -> list[Path]:
    slide_paths = []
    results = numbers["results"]
    finished = int(results["is_finished"].sum()) if not results.empty else 0
    scheduled = int(results["is_scheduled"].sum()) if not results.empty else 0
    total = len(results)
    augmented_rows = numbers["augmented_metrics"].get("training_rows", 0)
    augmented_accuracy = numbers["augmented_metrics"].get("eval_accuracy", 0) * 100
    live_rows = numbers["live_metrics"].get("training_rows", 0)
    live_accuracy = numbers["live_metrics"].get("eval_accuracy", 0) * 100
    score_eval = numbers["score_metrics"].get("eval_metrics", {})

    def save(canvas, index):
        path = ASSET_DIR / f"slide_{index}.png"
        canvas.convert("RGB").save(path, quality=95)
        slide_paths.append(path)

    canvas = Image.new("RGBA", (IMG_W, IMG_H), rgb(NAVY))
    decorate(canvas, dark=True)
    draw = ImageDraw.Draw(canvas)
    draw_header(draw, "FIFA World Cup 2026", "Football Match Prediction", dark=True)
    draw_text(
        draw,
        (66, 230),
        "A live-updating machine learning workflow for match outcomes, knockout advancement, tournament simulations, and scoreline probabilities.",
        38,
        rgb(WHITE),
        False,
        1040,
        line_gap=10,
    )
    card(draw, (72, 520, 362, 650), "ESPN fixtures", f"{total}", rgb(CYAN), value_size=42)
    card(draw, (392, 520, 682, 650), "Finished", f"{finished}", rgb(GREEN), value_size=42)
    card(draw, (712, 520, 1002, 650), "Scheduled", f"{scheduled}", rgb(BLUE), value_size=42)
    draw_text(
        draw,
        (66, 720),
        "Python | pandas | XGBoost | Airflow | ESPN scraping | Monte Carlo simulation | React + FastAPI",
        28,
        (208, 213, 221),
        False,
        1120,
    )
    save(canvas, 1)

    canvas = Image.new("RGBA", (IMG_W, IMG_H), rgb(PAPER))
    decorate(canvas)
    draw = ImageDraw.Draw(canvas)
    draw_header(draw, "Why this project matters", "Project Goal")
    bullets = [
        "Predict match outcomes for World Cup 2026 fixtures.",
        "Update the dataset as real ESPN results arrive.",
        "Simulate the full tournament path, not only one match.",
        "Keep original, augmented, live-result, and score models separate.",
        "Expose everything in an interactive React + FastAPI dashboard.",
    ]
    for i, item in enumerate(bullets):
        bullet(draw, 90, 260 + i * 86, item, max_width=630, size=25)
    rounded(draw, (880, 240, 1460, 610), rgb(WHITE), rgb(LINE), radius=14)
    paste_image(canvas, assets["snapshot"], (910, 270, 1430, 580))
    draw_text(draw, (900, 652), "The latest ESPN snapshot feeds both training and dashboard views.", 24, rgb(MUTED), False, 520)
    save(canvas, 2)

    canvas = Image.new("RGBA", (IMG_W, IMG_H), rgb(PAPER))
    decorate(canvas)
    draw = ImageDraw.Draw(canvas)
    draw_header(draw, "From raw match data to model outputs", "Data Pipeline")
    nodes = [
        ("Scrape", "ESPN live fixtures and results", rgb(CYAN)),
        ("Transform", "Clean teams, ELO, recent form", rgb(BLUE)),
        ("Train", "Augmented, live, and score models", rgb(PURPLE)),
        ("Simulate", "Monte Carlo tournament runs", rgb(AMBER)),
        ("Serve", "React + FastAPI app", rgb(GREEN)),
    ]
    x = 72
    for index, (label, value, color) in enumerate(nodes):
        rounded(draw, (x, 278, x + 250, 430), rgb(WHITE), rgb(LINE), radius=12)
        draw.rectangle((x, 278, x + 250, 292), fill=color)
        draw_text(draw, (x + 22, 315), label, 22, rgb(INK), True)
        draw_text(draw, (x + 22, 354), value, 17, rgb(MUTED), False, 205)
        if index < len(nodes) - 1:
            draw.line((x + 268, 354, x + 316, 354), fill=rgb(LINE), width=5)
            draw.polygon([(x + 316, 354), (x + 300, 342), (x + 300, 366)], fill=rgb(LINE))
        x += 308
    rounded(draw, (96, 555, 720, 815), rgb(WHITE), rgb(LINE), radius=14)
    draw_text(draw, (130, 590), "Airflow orchestration", 32, rgb(INK), True)
    for i, item in enumerate(
        [
            "Hourly ESPN refresh",
            "Augmented training branch",
            "Live-result-only training branch",
            "Versioned exact-score branch",
        ]
    ):
        bullet(draw, 132, 642 + i * 42, item, rgb(MUTED), max_width=500, size=18)
    rounded(draw, (840, 555, 1420, 780), rgb(WHITE), rgb(LINE), radius=14)
    draw_text(draw, (874, 590), "Versioning rule", 32, rgb(INK), True)
    draw_text(
        draw,
        (874, 650),
        "New artifacts are saved in timestamped folders, so the cloned baseline and older simulations remain available.",
        22,
        rgb(MUTED),
        False,
        480,
    )
    save(canvas, 3)

    canvas = Image.new("RGBA", (IMG_W, IMG_H), rgb(PAPER))
    decorate(canvas)
    draw = ImageDraw.Draw(canvas)
    draw_header(draw, "Separate models for separate questions", "Modeling Strategy")
    rounded(draw, (70, 230, 780, 695), rgb(WHITE), rgb(LINE), radius=14)
    paste_image(canvas, assets["metrics"], (105, 250, 750, 670))
    cards = [
        ("Augmented result model", f"{augmented_rows:,} rows | {augmented_accuracy:.1f}% eval accuracy"),
        ("Live result model", f"{live_rows:,} finished 2026 rows | {live_accuracy:.1f}% eval accuracy"),
        (
            "Exact score model",
            f"Top-3 scoreline hit rate: {score_eval.get('top3_scoreline_accuracy', 0) * 100:.1f}%",
        ),
    ]
    for i, (title, body) in enumerate(cards):
        top = 250 + i * 150
        rounded(draw, (850, top, 1420, top + 118), rgb(WHITE), rgb(LINE), radius=12)
        draw_text(draw, (882, top + 22), title, 25, rgb(INK), True)
        draw_text(draw, (882, top + 66), body, 20, rgb(MUTED), False, 470)
    draw_text(
        draw,
        (850, 730),
        "Exact scores are shown as probabilities, not certainties.",
        24,
        rgb(MUTED),
        False,
        600,
    )
    save(canvas, 4)

    canvas = Image.new("RGBA", (IMG_W, IMG_H), rgb(PAPER))
    decorate(canvas)
    draw = ImageDraw.Draw(canvas)
    draw_header(draw, "How the final product is used", "Dashboard and Results")
    rounded(draw, (70, 228, 790, 695), rgb(WHITE), rgb(LINE), radius=14)
    paste_image(canvas, assets["top"], (105, 250, 760, 670))
    rounded(draw, (870, 230, 1420, 520), rgb(WHITE), rgb(LINE), radius=14)
    draw_text(draw, (904, 270), "React + FastAPI app", 34, rgb(INK), True)
    for i, item in enumerate(
        [
            "Compare baseline, augmented, and live-result models.",
            "View tournament simulation probabilities.",
            "Inspect Round of 32 advancers and likely scorelines.",
            "Filter teams, statuses, fixtures, and rankings.",
        ]
    ):
        bullet(draw, 908, 340 + i * 42, item, rgb(MUTED), max_width=430, size=18)
    rounded(draw, (870, 570, 1420, 750), rgb(WHITE), rgb(LINE), radius=14)
    draw_text(draw, (904, 605), "Next steps", 34, rgb(INK), True)
    draw_text(
        draw,
        (904, 660),
        "Collect richer live match stats, improve calibration, and add model comparison reports for each retraining run.",
        20,
        rgb(MUTED),
        False,
        450,
    )
    draw_text(draw, (66, 835), "Branch: jon-clean | Dashboard: React + FastAPI | Scheduler: Airflow", 22, rgb(MUTED), False)
    save(canvas, 5)

    return slide_paths


def start_libreoffice():
    profile_dir = Path("/tmp/worldcup-presentation-lo-profile")
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "libreoffice",
        "--headless",
        "--invisible",
        "--nodefault",
        "--nofirststartwizard",
        f"-env:UserInstallation={file_url(profile_dir)}",
        "--accept=socket,host=localhost,port=2002;urp;StarOffice.ComponentContext",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def connect_libreoffice():
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver",
        local_ctx,
    )
    for _ in range(40):
        try:
            return resolver.resolve(
                "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
            )
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Could not connect to LibreOffice.")


def set_slide_size(doc, page):
    try:
        doc.DrawPage.Width = SLIDE_W
        doc.DrawPage.Height = SLIDE_H
    except Exception:
        pass
    try:
        page.Width = SLIDE_W
        page.Height = SLIDE_H
    except Exception:
        pass


def add_shape(doc, page, service, x, y, w, h, **properties):
    shape = doc.createInstance(service)
    shape.Position = Point(x, y)
    shape.Size = Size(w, h)
    for key, value in properties.items():
        try:
            setattr(shape, key, value)
        except Exception:
            pass
    page.add(shape)
    return shape


def rect(doc, page, x, y, w, h, color, line_color=None):
    shape = add_shape(
        doc,
        page,
        "com.sun.star.drawing.RectangleShape",
        x,
        y,
        w,
        h,
        FillColor=color,
        LineColor=line_color if line_color is not None else color,
    )
    return shape


def text(doc, page, x, y, w, h, value, size=24, color=INK, bold=False):
    shape = add_shape(
        doc,
        page,
        "com.sun.star.drawing.TextShape",
        x,
        y,
        w,
        h,
        FillTransparence=100,
        LineTransparence=100,
    )
    shape.String = value
    cursor = shape.createTextCursor()
    cursor.gotoStart(False)
    cursor.gotoEnd(True)
    cursor.CharHeight = float(size)
    cursor.CharColor = color
    cursor.CharFontName = "Aptos"
    cursor.CharWeight = 150 if bold else 100
    return shape


def image(doc, page, path, x, y, w, h):
    return add_shape(
        doc,
        page,
        "com.sun.star.drawing.GraphicObjectShape",
        x,
        y,
        w,
        h,
        GraphicURL=file_url(path),
    )


def pill(doc, page, x, y, w, h, label, value, color):
    rect(doc, page, x, y, w, h, WHITE, LINE)
    rect(doc, page, x, y, 420, h, color, color)
    text(doc, page, x + 650, y + 240, w - 900, 420, label, 15, MUTED, False)
    text(doc, page, x + 650, y + 720, w - 900, 650, value, 28, INK, True)


def title_bar(doc, page, title, kicker):
    text(doc, page, 950, 650, 18000, 550, kicker.upper(), 12, CYAN, True)
    text(doc, page, 950, 1150, 22000, 900, title, 34, WHITE, True)
    rect(doc, page, 0, 0, 360, SLIDE_H, CYAN, CYAN)


def add_background(doc, page, dark=False):
    rect(doc, page, 0, 0, SLIDE_W, SLIDE_H, NAVY if dark else PAPER)
    rect(doc, page, SLIDE_W - 5200, 0, 5200, SLIDE_H, 0xE0F2FE if not dark else 0x172554)
    rect(doc, page, SLIDE_W - 4400, 1300, 3300, 3300, CYAN if not dark else BLUE, CYAN)
    rect(doc, page, SLIDE_W - 3500, 2400, 2600, 2600, BLUE if not dark else PURPLE, BLUE)


def slide_title(doc, page, title, subtitle):
    text(doc, page, 950, 700, 18000, 560, subtitle.upper(), 12, BLUE, True)
    text(doc, page, 950, 1200, 20000, 900, title, 32, INK, True)


def bullet_list(doc, page, x, y, items, color=INK):
    current_y = y
    for item in items:
        rect(doc, page, x, current_y + 110, 150, 150, BLUE, BLUE)
        text(doc, page, x + 380, current_y, 10200, 520, item, 18, color, False)
        current_y += 780


def pipeline_node(doc, page, x, y, label, value, color):
    rect(doc, page, x, y, 3600, 1500, WHITE, LINE)
    rect(doc, page, x, y, 3600, 170, color, color)
    text(doc, page, x + 250, y + 360, 3000, 400, label, 14, MUTED, True)
    text(doc, page, x + 250, y + 760, 3000, 520, value, 19, INK, True)


def arrow(doc, page, x, y, w):
    rect(doc, page, x, y + 120, w, 80, LINE, LINE)
    rect(doc, page, x + w - 120, y, 260, 320, LINE, LINE)


def make_deck(slide_paths):
    process = start_libreoffice()
    try:
        ctx = connect_libreoffice()
        smgr = ctx.ServiceManager
        desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
        pages = doc.getDrawPages()
        while pages.getCount() < 5:
            pages.insertNewByIndex(pages.getCount())
        while pages.getCount() > 5:
            pages.remove(pages.getByIndex(pages.getCount() - 1))

        for i in range(5):
            set_slide_size(doc, pages.getByIndex(i))

        for index, slide_path in enumerate(slide_paths):
            page = pages.getByIndex(index)
            image(doc, page, slide_path, 0, 0, SLIDE_W, SLIDE_H)

        PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
        store_props = (
            prop("FilterName", "Impress MS PowerPoint 2007 XML"),
            prop("Overwrite", True),
        )
        doc.storeAsURL(file_url(OUTPUT_PATH), store_props)
        doc.close(True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-worldcup-presentation")
    numbers = load_project_numbers()
    assets = save_charts(numbers)
    slide_paths = render_slide_images(numbers, assets)
    make_deck(slide_paths)
    print(f"Saved presentation: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
