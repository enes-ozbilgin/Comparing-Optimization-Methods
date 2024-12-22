import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
gd_data = pd.read_csv('C:\\Users\\ENES\\Desktop\\csv_logs\\gd_time_log_run5.csv')
sgd_data = pd.read_csv('C:\\Users\\ENES\\Desktop\\csv_logs\\sgd_time_log_run5.csv')
adam_data = pd.read_csv('C:\\Users\\ENES\\Desktop\\csv_logs\\adam_time_log_run5.csv')

# Print the column names of each dataset
print("GD Data columns:", gd_data.columns)
print("SGD Data columns:", sgd_data.columns)
print("Adam Data columns:", adam_data.columns)

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot GD
plt.plot(gd_data['Time'], gd_data['Loss'], label='GD')

# Plot SGD
plt.plot(sgd_data['Time'], sgd_data['Loss'], label='SGD')

# Plot Adam
plt.plot(adam_data['Time'], adam_data['Loss'], label='Adam')

# Adding labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Time vs Loss for SGD, Adam, and GD')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
