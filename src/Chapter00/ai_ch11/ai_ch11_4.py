#
# Probabilistic reasoning for sequential data: Slicer
#
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from ai_utils import read_data 

# Load input data
index = 2
data = read_data('data/ai_ch11/data_2D.txt', index)

# Plot data with year-level granularity 
start = '2003'
end = '2011'
plt.figure()
data[start:end].plot()
plt.title('Input data from ' + start + ' to ' + end)

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai11_plot_url1 = b64encode(img.getvalue()).decode('ascii')

# Plot data with month-level granularity 
start = '1998-2'
end = '2006-7'
plt.figure()
data[start:end].plot()
plt.title('Input data from ' + start + ' to ' + end)

#plt.show()
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai11_plot_url2 = b64encode(img.getvalue()).decode('ascii')