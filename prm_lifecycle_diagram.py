"""Render an editable PRM overview as SVG, PDF, and high-resolution PNG.

Usage: python prm_lifecycle_diagram.py --output-dir figures --dpi 300
Requires matplotlib. All geometry and illustrations are vector primitives.
Angles use Matplotlib's convention: zero points right, positive is CCW.
"""
from pathlib import Path
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyArrowPatch, FancyBboxPatch, Arc

INK = "#192B38"
CENTER = (0, 0)


def point(radius, angle):
    a = math.radians(angle)
    return radius * math.cos(a), radius * math.sin(a)


def band(ax, start, end, inner, outer, color, edge):
    ax.add_patch(Wedge(CENTER, outer, start, end, width=outer-inner,
                       facecolor=color, edgecolor=edge, linewidth=1.1))


def tangent_label(ax, text, radius, angle, size=12, weight="normal"):
    x, y = point(radius, angle)
    rotation = angle - 90
    if rotation < -90:
        rotation += 180
    if rotation > 90:
        rotation -= 180
    ax.text(x, y, text, ha="center", va="center", rotation=rotation,
            rotation_mode="anchor", fontsize=size, fontweight=weight,
            color=INK, linespacing=1.15)


def curved_label(ax, text, radius, start, end, size=17):
    """Place letters along an arc with proportional spacing, upright at center."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    prop = FontProperties(family="DejaVu Sans", weight="bold", size=size)
    widths = [TextPath((0, 0), c, prop=prop).get_extents().width
              if c != " " else size*.35 for c in text]
    widths = [max(w, size*.17) + size*.10 for w in widths]
    total = sum(widths)
    cursor = 0
    for char, width in zip(text, widths):
        angle = start + (end-start)*(cursor+width/2)/total
        x, y = point(radius, angle)
        rot = angle-90 if end < start else angle+90
        ax.text(x, y, char, ha="center", va="center", rotation=rot,
                rotation_mode="anchor", fontproperties=prop, color=INK)
        cursor += width


def app_icon(ax, kind, x, y, color):
    """Small consistent line illustrations, without platform-dependent emoji."""
    def line(xs, ys):
        ax.plot([x+v for v in xs], [y+v for v in ys], color=color,
                lw=1.3, solid_capstyle="round")
    def box(dx, dy, w, h):
        ax.add_patch(FancyBboxPatch((x+dx, y+dy), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            fill=False, edgecolor=color, linewidth=1.2))
    if kind in ("Robotics", "Agent"):
        box(-.15,-.11,.30,.22)
        line([0,0], [.12,.20])
        for dx in [-.07,.07]:
            ax.add_patch(Circle((x+dx,y+.02), .018, color=color))
        line([-.05,.05],[-.06,-.06])
    elif kind == "Multimodal":
        box(-.17,-.13,.26,.23)
        line([-.14,-.05,.06],[-.08,.03,-.08])
        line([.14,.14,.21,.21],[.10,-.02,.04,-.08])
    elif kind == "Industry":
        line([-.17,-.17,-.03,-.03,.12,.12,.19,.19,-.17],
             [-.12,.07,-.01,.08,-.01,.17,.17,-.12,-.12])
    elif kind == "Math":
        ax.text(x,y,r"$\Sigma$",fontsize=24,color=color,ha="center",va="center")
    elif kind == "Text":
        box(-.12,-.15,.24,.30)
        for dy in [-.07,0,.07]:
            line([-.07,.07],[dy,dy])
    elif kind == "Code":
        ax.text(x,y,"</>",fontsize=15,color=color,ha="center",va="center",
                fontfamily="DejaVu Sans Mono")


def draw(output_dir, dpi):
    plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none",
                         "pdf.fonttype":42, "savefig.facecolor":"white"})
    fig, ax = plt.subplots(figsize=(12.4, 8.4))
    ax.set_aspect("equal")
    ax.set_xlim(-3.75, 6.0)
    ax.set_ylim(-3.7, 3.7)
    ax.axis("off")
    fig.subplots_adjust(left=.025,right=.985,bottom=.025,top=.975)

    # Main lifecycle: each label occupies a dedicated band.
    band(ax, 78,178,1.32,2.43,"#59DDB3","#168767")
    band(ax, 187,284,1.32,2.43,"#BDDCEF","#4B8CB5")
    band(ax,-64,62,1.32,2.43,"#FC777D","#CC4855")
    curved_label(ax,"PRM TRAINING",1.91,163,91,17)
    curved_label(ax,"DATA PROCESS",1.91,202,270,17)
    curved_label(ax,"PRM USAGE",1.91,43,-44,17)

    # Training and data subcategories retain the original grouping.
    tabs = [
        (163,186,"Discriminative","#C3EEDD","#499981"),
        (137,160,"Generative","#AEE7CC","#499981"),
        (111,134,"Implicit","#98DDBB","#499981"),
        (78,108,"Architectures","#8FD2C7","#499981"),
        (188,218,"Human","#B5C6ED","#687CAD"),
        (220,251,"Automated","#A1B1EE","#687CAD"),
        (253,284,"Semi-auto","#C1AAEE","#8570AE"),
    ]
    for start,end,label,color,edge in tabs:
        band(ax,start,end,2.62,3.27,color,edge)
        tangent_label(ax,label,2.945,(start+end)/2,11.5)

    band(ax,3,62,2.62,3.27,"#FFC1BA","#D47E78")
    band(ax,-64,-1,2.62,3.27,"#F3BBD6","#C779A2")
    tangent_label(ax,"Test-Time\nScaling",2.945,32.5,13,"bold")
    tangent_label(ax,"RL for Policy\nLearning",2.945,-32.5,13,"bold")

    # Central neural graph and circular update arrow.
    ax.add_patch(Circle((0,0),1.03,facecolor="#FAFBFC",edgecolor="#DCE3E7",lw=1))
    nodes=[(-.57,.25),(-.29,.61),(.13,.65),(.51,.32),(.58,-.18),
           (.21,-.55),(-.32,-.53),(-.62,-.15),(0,.14),(-.05,-.26)]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),
           (0,8),(1,8),(2,8),(3,8),(4,9),(5,9),(6,9),(7,9),(8,9)]
    for i,j in edges:
        ax.plot([nodes[i][0],nodes[j][0]],[nodes[i][1],nodes[j][1]],
                color="#D6DFE4",lw=1.3,zorder=2)
    for x,y in nodes:
        ax.add_patch(Circle((x,y),.055,facecolor="white",edgecolor="#A9BAC4",lw=1,zorder=3))
    ax.add_patch(Arc((0,0),2.34,2.34,theta1=-35,theta2=278,
                     color="#C98747",lw=9,capstyle="round"))
    ax.add_patch(FancyArrowPatch(point(1.17,-28),point(1.17,-45),
                    connectionstyle="arc3,rad=-0.1",arrowstyle="-|>",
                    mutation_scale=28,lw=3,color="#C98747",zorder=5))
    ax.text(0,0,"PRM",fontsize=32,fontweight="bold",ha="center",va="center",
            color=INK,bbox=dict(facecolor="#FAFBFC",edgecolor="none",pad=1.5),zorder=6)

    # Downstream applications: consistent cards and line icons.
    ax.plot([3.65,3.65],[-2.91,2.91],color="#E7D4AA",lw=1.3)
    ax.text(3.50,0,"DOWNSTREAM APPLICATIONS",rotation=-90,
            fontsize=12,fontweight="bold",ha="center",va="center",color="#806A3E")
    names=["Robotics","Multimodal","Industry","Math","Text","Code","Agent"]
    for i,name in enumerate(names):
        y=2.64-i*.88
        ax.add_patch(FancyBboxPatch((3.94,y-.32),1.89,.64,
            boxstyle="round,pad=0.02,rounding_size=0.065",facecolor="#FFF7E5",
            edgecolor="#D8BC7F",lw=1))
        app_icon(ax,name,4.25,y,"#9B7836")
        ax.text(4.62,y,name,fontsize=12.5,ha="left",va="center",color=INK)

    output_dir.mkdir(parents=True,exist_ok=True)
    for suffix in ("svg","pdf","png"):
        destination=output_dir/f"prm_lifecycle_diagram.{suffix}"
        fig.savefig(destination,dpi=dpi)
        print(destination.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("figures"))
    parser.add_argument("--dpi",type=int,default=300)
    args=parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    draw(args.output_dir,args.dpi)
