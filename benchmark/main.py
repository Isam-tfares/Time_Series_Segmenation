import ruptures as rpt
import matplotlib.pyplot as plt
import pandas as pd

# df=pd.read_csv(r"C:\Users\hp\Desktop\Time_Series_Segmenation\datasets\SKAB_data\valve1\0.csv",sep=";")
# ts=df.Accelerometer1RMS.values
# # print(df.columns)
# true_bkps=df[df.changepoint==1.0].index
# # print(ts.shape,true_bkps)



# model = "l2"  # Choose a cost function ("l1", "l2", etc.)
# window = Window(width=40, model=model)
# window.fit(ts)
# breakpoints = window.predict(n_bkps=len(true_bkps)) 


# plt.figure()
# plt.step(np.arange(len(ts)), ts)  # Plot signal as steps

# for bp in breakpoints:
#   plt.axvline(x=bp, color='r', linestyle='dashed', linewidth=1)
# for bp in true_bkps:
#   plt.axvline(x=bp, color='green', linestyle='dashed', linewidth=1)
# plt.xlabel('Time')
# plt.ylabel('Signal Value')
# plt.title('Signal with Predicted Change Points')
# plt.grid(True)

# plt.show()
