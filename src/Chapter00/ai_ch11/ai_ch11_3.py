#
# Probabilistic reasoning for sequential data: Operator
#
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from ai_utils import read_data 

# Input filename
input_file = 'data/ai_ch11/data_2D.txt'

# Load data
x1 = read_data(input_file, 2)
x2 = read_data(input_file, 3)

# Create pandas dataframe for slicing
data = pd.DataFrame({'dim1': x1, 'dim2': x2})

# Plot data
start = '1968'
end = '1975'
data[start:end].plot()
plt.title('Data overlapped on top of each other')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai11_plot_url1 = b64encode(img.getvalue()).decode('ascii')

# Filtering using conditions
# - 'dim1' is smaller than a certain threshold
# - 'dim2' is greater than a certain threshold
data[(data['dim1'] < 45) & (data['dim2'] > 30)].plot()
plt.title('dim1 < 45 and dim2 > 30')

img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai11_plot_url2 = b64encode(img.getvalue()).decode('ascii')

# Adding two dataframes 
plt.figure()
diff = data[start:end]['dim1'] + data[start:end]['dim2']
diff.plot()
plt.title('Summation (dim1 + dim2)')

#plt.show()
img = BytesIO()
plt.savefig(img, dpi=300)
plt.close()
img.seek(0)
    
ai11_plot_url3 = b64encode(img.getvalue()).decode('ascii')