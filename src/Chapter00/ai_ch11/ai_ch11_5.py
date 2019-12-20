#
# Probabilistic reasoning for sequential data: Stats Extractor
#
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from base64 import b64encode
from ai_utils import read_data 

# Input filename
input_file = 'data/ai_ch11/data_2D.txt'

# Load input data in time series format
x1 = read_data(input_file, 2)
x2 = read_data(input_file, 3)

# Create pandas dataframe for slicing
data = pd.DataFrame({'dim1': x1, 'dim2': x2})

# Extract max and min values
print('\nMaximum values for each dimension:')
print(data.max())
ai11_5_url1 = data.max()
print('\nMinimum values for each dimension:')
print(data.min())
ai11_5_url2 = data.min()

# Extract overall mean and row-wise mean values
print('\nOverall mean:')
print(data.mean())
ai11_5_url3 = data.mean()
print('\nRow-wise mean:')
print(data.mean(1)[:12])
ai11_5_url4 = data.mean(1)[:12]

# Plot the rolling mean using a window size of 24
data.rolling(center=False, window=24).mean().plot()
plt.title('Rolling mean')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai11_plot_url1 = b64encode(img.getvalue()).decode('ascii')

# Extract correlation coefficients
print('\nCorrelation coefficients:\n', data.corr())
ai11_5_url5 = data.corr()

# Plot rolling correlation using a window size of 60
plt.figure()
plt.title('Rolling correlation')
data['dim1'].rolling(window=60).corr(other=data['dim2']).plot()

#plt.show()
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai11_plot_url2 = b64encode(img.getvalue()).decode('ascii')