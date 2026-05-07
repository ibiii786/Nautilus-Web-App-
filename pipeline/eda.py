"""
Nautilus — Exploratory Data Analysis
=====================================
Step 3: EDA with deep black and bioluminescent visualizations.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

DARK_BG = "#000000"
CARD_BG = "#0a120f"
AC1, AC2, AC3, AC4, AC5 = "#00ffcc", "#39ff14", "#00bfff", "#ccff00", "#ffffff"
TEXT_COL = "#e0f2f1"
GRID_COL = "#11221c"
PAL = [AC1, AC2, AC3, AC4, AC5, "#00ffaa", "#aaff00", "#00aaff", "#ccffaa", "#e0ffe0"]
OCEAN_CMAP = LinearSegmentedColormap.from_list("ocean", ["#000000","#002211","#00ffcc","#39ff14","#ffffff"])

def _theme():
    plt.rcParams.update({"figure.facecolor":DARK_BG,"axes.facecolor":CARD_BG,"axes.edgecolor":GRID_COL,
        "axes.labelcolor":TEXT_COL,"text.color":TEXT_COL,"xtick.color":TEXT_COL,"ytick.color":TEXT_COL,
        "grid.color":GRID_COL,"grid.alpha":0.3,"font.size":11,"axes.titlesize":14,"axes.titleweight":"bold",
        "legend.facecolor":CARD_BG,"legend.edgecolor":GRID_COL,"legend.fontsize":9})

def generate_all_plots(stats_df, objects_df, plots_dir):
    _theme(); os.makedirs(plots_dir, exist_ok=True)
    plots = [
        _p1_category(objects_df, plots_dir),
        _p2_dimensions(stats_df, plots_dir),
        _p3_colors(stats_df, plots_dir),
        _p4_bbox(objects_df, plots_dir),
        _p5_heatmap(objects_df, stats_df, plots_dir),
        _p6_brightness(stats_df, plots_dir),
        _p7_objects(stats_df, plots_dir),
        _p8_corr(stats_df, plots_dir),
    ]
    plt.close("all")
    return plots

def _p1_category(df, d):
    fig, ax = plt.subplots(figsize=(10,6))
    if len(df)>0:
        c = df["class"].value_counts()
        bars = ax.barh(c.index, c.values, color=PAL[:len(c)], height=0.7)
        for b,v in zip(bars, c.values):
            ax.text(v+max(c.values)*0.01, b.get_y()+b.get_height()/2, f"{v:,}", va="center", fontsize=10, color=TEXT_COL, fontweight="bold")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.2)
    ax.set_title("🐠 Species / Category Distribution", fontsize=16, pad=15)
    ax.set_xlabel("Number of Annotations")
    plt.tight_layout(); p=os.path.join(d,"01_category_distribution.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig)
    return {"name":"Category Distribution","file":"01_category_distribution.png","path":p,"description":"Distribution of annotated marine species across the dataset"}

def _p2_dimensions(df, d):
    fig, ax = plt.subplots(figsize=(10,6))
    if len(df)>0 and "width" in df.columns:
        sc = ax.scatter(df["width"],df["height"],c=df["mean_brightness"],cmap=OCEAN_CMAP,alpha=0.6,s=15)
        cb = plt.colorbar(sc,ax=ax,pad=0.02); cb.set_label("Mean Brightness",color=TEXT_COL)
        cb.ax.yaxis.set_tick_params(color=TEXT_COL); plt.setp(plt.getp(cb.ax.axes,'yticklabels'),color=TEXT_COL)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)"); ax.set_title("📐 Image Dimension Distribution",fontsize=16,pad=15); ax.grid(True,alpha=0.2)
    plt.tight_layout(); p=os.path.join(d,"02_dimension_distribution.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig)
    return {"name":"Dimension Distribution","file":"02_dimension_distribution.png","path":p,"description":"Scatter plot of image dimensions colored by brightness"}

def _p3_colors(df, d):
    fig, axes = plt.subplots(1,3,figsize=(14,5))
    for ax,(col,t,clr) in zip(axes,[("mean_red","Red","#ff4444"),("mean_green","Green","#44ff88"),("mean_blue","Blue","#4488ff")]):
        if len(df)>0 and col in df.columns:
            ax.hist(df[col].dropna(),bins=40,color=clr,alpha=0.7); m=df[col].mean()
            ax.axvline(m,color="#fff",ls="--",lw=1.5,label=f"Mean:{m:.1f}"); ax.legend()
        ax.set_title(t,fontsize=13); ax.set_xlabel("Intensity"); ax.grid(True,alpha=0.2)
    fig.suptitle("🎨 Color Channel Distribution (RGB)",fontsize=16,fontweight="bold",y=1.02)
    plt.tight_layout(); p=os.path.join(d,"03_color_channels.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig)
    return {"name":"Color Channels","file":"03_color_channels.png","path":p,"description":"RGB distributions showing underwater color cast"}

def _p4_bbox(df, d):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    if len(df)>0:
        axes[0].hist(df["bbox_area"].clip(upper=df["bbox_area"].quantile(0.95)),bins=50,color=AC1,alpha=0.7)
        axes[0].set_title("Box Area Distribution",fontsize=13); axes[0].set_xlabel("Area (px²)"); axes[0].grid(True,alpha=0.2)
        s=df.sample(min(2000,len(df)),random_state=42)
        axes[1].scatter(s["bbox_width"],s["bbox_height"],c=AC2,alpha=0.3,s=8)
        axes[1].set_title("Width vs Height",fontsize=13); axes[1].set_xlabel("Width"); axes[1].set_ylabel("Height"); axes[1].grid(True,alpha=0.2)
    fig.suptitle("📏 Bounding Box Size Analysis",fontsize=16,fontweight="bold",y=1.02)
    plt.tight_layout(); p=os.path.join(d,"04_bbox_sizes.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig)
    return {"name":"Bounding Box Sizes","file":"04_bbox_sizes.png","path":p,"description":"Object size distributions and aspect ratios"}

def _p5_heatmap(odf, sdf, d):
    fig, ax = plt.subplots(figsize=(10,8))
    hm = np.zeros((64,64),dtype=np.float32)
    if len(odf)>0 and len(sdf)>0:
        for _,r in odf.iterrows():
            inf = sdf[sdf["filename"]==r["filename"]]
            if len(inf)==0: continue
            iw,ih = inf.iloc[0]["width"],inf.iloc[0]["height"]
            if iw==0 or ih==0: continue
            cx=int(np.clip(((r["xmin"]+r["xmax"])/2)/iw*63,0,63))
            cy=int(np.clip(((r["ymin"]+r["ymax"])/2)/ih*63,0,63))
            hm[cy,cx]+=1
        from scipy.ndimage import gaussian_filter
        hm = gaussian_filter(hm,sigma=3)
    im=ax.imshow(hm,cmap=OCEAN_CMAP,aspect="auto",interpolation="bilinear")
    cb=plt.colorbar(im,ax=ax,pad=0.02); cb.set_label("Density",color=TEXT_COL)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL); plt.setp(plt.getp(cb.ax.axes,'yticklabels'),color=TEXT_COL)
    ax.set_title("🔥 Object Location Heatmap",fontsize=16,pad=15)
    ax.set_xticks([0,32,63]); ax.set_xticklabels(["Left","Center","Right"])
    ax.set_yticks([0,32,63]); ax.set_yticklabels(["Top","Center","Bottom"])
    plt.tight_layout(); p=os.path.join(d,"05_annotation_heatmap.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig)
    return {"name":"Object Heatmap","file":"05_annotation_heatmap.png","path":p,"description":"Spatial density showing where marine life appears"}

def _p6_brightness(df, d):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    if len(df)>0:
        parts=axes[0].violinplot([df["mean_brightness"].dropna()],positions=[1],showmeans=True,showmedians=True)
        for pc in parts["bodies"]: pc.set_facecolor(AC1); pc.set_alpha(0.6)
        parts["cmeans"].set_color(AC5); parts["cmedians"].set_color(AC4)
        axes[0].set_title("Brightness",fontsize=13); axes[0].set_ylabel("Mean (0-255)"); axes[0].set_xticks([1]); axes[0].set_xticklabels(["All"]); axes[0].grid(True,alpha=0.2)
        bp=axes[1].boxplot([df["mean_red"].dropna(),df["mean_green"].dropna(),df["mean_blue"].dropna()],labels=["Red","Green","Blue"],patch_artist=True,widths=0.5)
        for p2,c in zip(bp["boxes"],["#ff4444","#44ff88","#4488ff"]): p2.set_facecolor(c); p2.set_alpha(0.6)
        for el in ["whiskers","caps","medians"]: plt.setp(bp[el],color=TEXT_COL)
        axes[1].set_title("Channel Comparison",fontsize=13); axes[1].grid(True,alpha=0.2)
    fig.suptitle("💡 Brightness & Color Analysis",fontsize=16,fontweight="bold",y=1.02)
    plt.tight_layout(); p=os.path.join(d,"06_brightness.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig)
    return {"name":"Brightness Analysis","file":"06_brightness.png","path":p,"description":"Brightness and RGB channel intensity comparison"}

def _p7_objects(df, d):
    fig, ax = plt.subplots(figsize=(10,6))
    if len(df)>0:
        c=df["num_objects"]
        ax.hist(c,bins=range(0,int(c.max())+2),color=AC3,alpha=0.7,edgecolor=DARK_BG)
        ax.axvline(c.mean(),color=AC5,ls="--",lw=2,label=f"Mean: {c.mean():.1f}")
        ax.legend(fontsize=11)
    ax.set_xlabel("Number of Objects"); ax.set_ylabel("Images"); ax.set_title("📦 Objects per Image",fontsize=16,pad=15); ax.grid(True,alpha=0.2)
    plt.tight_layout(); p=os.path.join(d,"07_objects_per_image.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig)
    return {"name":"Objects per Image","file":"07_objects_per_image.png","path":p,"description":"Distribution of objects annotated per image"}

def _p8_corr(df, d):
    fig, ax = plt.subplots(figsize=(10,8))
    if len(df)>0:
        cols=[c for c in ["width","height","num_objects","mean_brightness","mean_red","mean_green","mean_blue"] if c in df.columns]
        if len(cols)>=3:
            corr=df[cols].corr()
            names={"width":"Width","height":"Height","num_objects":"Objects","mean_brightness":"Brightness","mean_red":"Red","mean_green":"Green","mean_blue":"Blue"}
            corr.rename(columns=names,index=names,inplace=True)
            mask=np.triu(np.ones_like(corr,dtype=bool),k=1)
            sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap=OCEAN_CMAP,center=0,square=True,linewidths=1,linecolor=DARK_BG,cbar_kws={"shrink":0.8},ax=ax,annot_kws={"fontsize":10,"fontweight":"bold"})
    ax.set_title("🔗 Feature Correlation Matrix",fontsize=16,pad=15)
    plt.tight_layout(); p=os.path.join(d,"08_correlation.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig)
    return {"name":"Feature Correlation","file":"08_correlation.png","path":p,"description":"Correlations between image properties and object counts"}

def derive_insights(stats_df, objects_df):
    insights = []
    if len(stats_df)>0:
        mr,mg,mb = stats_df["mean_red"].mean(),stats_df["mean_green"].mean(),stats_df["mean_blue"].mean()
        dom = "blue" if mb>mg and mb>mr else ("green" if mg>mr else "red")
        insights.append({"title":"🎨 Underwater Color Cast","text":f"Dominant {dom} channel (R={mr:.1f}, G={mg:.1f}, B={mb:.1f}) confirms underwater color distortion from wavelength-dependent light absorption, validating the need for color correction.","type":"color"})
        ao=stats_df["num_objects"].mean(); mx=stats_df["num_objects"].max(); emp=(stats_df["num_objects"]==0).sum()
        insights.append({"title":"📊 Object Density","text":f"Average {ao:.1f} objects/image (max {mx}). {emp} images unannotated. Varying complexity requires the model to handle both sparse and dense scenes.","type":"density"})
        ab=stats_df["mean_brightness"].mean(); sb=stats_df["mean_brightness"].std(); lv=(stats_df["mean_brightness"]<80).sum(); pct=lv/len(stats_df)*100
        insights.append({"title":"💡 Visibility Challenge","text":f"Mean brightness {ab:.1f}/255 (σ={sb:.1f}). {pct:.1f}% below low-visibility threshold (<80), confirming light absorption at depth. Enhancement is critical before detection.","type":"brightness"})
    if len(objects_df)>0:
        vc=objects_df["class"].value_counts(); most=vc.index[0]; least=vc.index[-1]; ratio=vc.iloc[0]/max(vc.iloc[-1],1)
        insights.append({"title":"⚖️ Category Imbalance","text":f"'{most}' most frequent ({vc.iloc[0]:,}×) vs '{least}' rarest ({vc.iloc[-1]:,}×) — {ratio:.1f}× imbalance. May need augmentation or class weighting for minority species.","type":"imbalance"})
    return insights
