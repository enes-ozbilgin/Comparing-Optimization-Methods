import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
gd_data = pd.read_csv("C:\\Users\\ENES\\Desktop\\csv_logs\\gd_accuracy_log_run5.csv")
sgd_data = pd.read_csv('C:\\Users\\ENES\\Desktop\\csv_logs\\sgd_accuracy_log_run5.csv')
adam_data = pd.read_csv('C:\\Users\\ENES\\Desktop\\csv_logs\\adam_accuracy_log_run5.csv')

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot GD
plt.plot(gd_data['Epoch'], gd_data['Accuracy'], label='GD')

# Plot SGD
plt.plot(sgd_data['Epoch'], sgd_data['Accuracy'], label='SGD')

# Plot Adam
plt.plot(adam_data['Epoch'], adam_data['Accuracy'], label='Adam')

# Adding labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy for GD, SGD, and ADAM')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
