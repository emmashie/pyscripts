import numpy as np 
import os
import glob

figpath = "/opt/data/delft/sfb_dfm_v2/runs/wy2013a/DFM_OUTPUT_wy2013a/validation_plots/usgs_transects/"
#figs = os.listdir(figpath)
figs = glob.glob("*.png")
f = open(figpath + "latex_fig.txt", "w")

for fig in figs:
	f.write("\\begin{figure}[h!] \n")
	f.write("\\centering \n")
	f.write("\\includegraphics[scale=0.5]{../wy2013a/validation_plots/usgs_transects/%s} \n" % (fig))
	f.write("\\caption{} \n")
	f.write("\\label{fig:%s} \n" % (fig[:-4]))
	f.write("\\end{figure} \n")
	f.write("\\FloatBarrier \n\n")

f.close()
