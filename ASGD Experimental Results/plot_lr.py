# matplotlib.rcParams['text.usetex'] = True
import os
import re
import csv

import matplotlib 
import matplotlib.pyplot as plt
cd = 'C:/Users/sjhuv/Desktop/Misc/ASGD Experimental Results'
os.chdir(cd)

val_record_1 = []

with open('small_lr_result_1AR-SGD-IBout_r0_n2.csv') as csv_file:
    #### alpha = 0.005
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip the first four lines indicating the state of current program
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            col_name = row
            line_count += 1
        # read in line every 4 lines
        elif line_count%4 == 3:
            single_epoch = {}
            single_epoch['epoch'] = row[col_name.index('Epoch')]
            single_epoch['Loss'] = row[col_name.index('Loss')]
            single_epoch['Prec@1'] = row[col_name.index('Prec@1')]
            single_epoch['Prec@5'] = row[col_name.index('Prec@5')]
            line_count += 1
            val_record_1.append(single_epoch)
        else:
            line_count += 1
            continue
            # print(f'\t{row[0]} is in the {row[1]} condition, and has disease {row[3]}.')
#            patient_record.append(single_patient)
    print(f'Processed {line_count} lines.')
    
    
val_record_2 = []

with open('small_lr_result_2AR-SGD-IBout_r0_n2.csv') as csv_file:
    #### alpha = 0.01
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip the first four lines indicating the state of current program
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            col_name = row
            line_count += 1
        # read in line every 4 lines
        elif line_count%4 == 3:
            single_epoch = {}
            single_epoch['epoch'] = row[col_name.index('Epoch')]
            single_epoch['Loss'] = row[col_name.index('Loss')]
            single_epoch['Prec@1'] = row[col_name.index('Prec@1')]
            single_epoch['Prec@5'] = row[col_name.index('Prec@5')]
            line_count += 1
            val_record_2.append(single_epoch)
        else:
            line_count += 1
            continue
            # print(f'\t{row[0]} is in the {row[1]} condition, and has disease {row[3]}.')
#            patient_record.append(single_patient)
    print(f'Processed {line_count} lines.')
    
    
    
val_record_3 = []

with open('small_lr_result_4AR-SGD-IBout_r0_n2.csv') as csv_file:
    #### alpha = 0.02
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip the first four lines indicating the state of current program
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            col_name = row
            line_count += 1
        # read in line every 4 lines
        elif line_count%4 == 3:
            single_epoch = {}
            single_epoch['epoch'] = row[col_name.index('Epoch')]
            single_epoch['Loss'] = row[col_name.index('Loss')]
            single_epoch['Prec@1'] = row[col_name.index('Prec@1')]
            single_epoch['Prec@5'] = row[col_name.index('Prec@5')]
            line_count += 1
            val_record_3.append(single_epoch)
        else:
            line_count += 1
            continue
            # print(f'\t{row[0]} is in the {row[1]} condition, and has disease {row[3]}.')
#            patient_record.append(single_patient)
    print(f'Processed {line_count} lines.')
    
    
val_record_4 = []

with open('large_lr_result_1AR-SGD-IBout_r0_n2.csv') as csv_file:
    #### alpha = 0.04
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip the first four lines indicating the state of current program
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            col_name = row
            line_count += 1
        # read in line every 4 lines
        elif line_count%4 == 3:
            single_epoch = {}
            single_epoch['epoch'] = row[col_name.index('Epoch')]
            single_epoch['Loss'] = row[col_name.index('Loss')]
            single_epoch['Prec@1'] = row[col_name.index('Prec@1')]
            single_epoch['Prec@5'] = row[col_name.index('Prec@5')]
            line_count += 1
            val_record_4.append(single_epoch)
        else:
            line_count += 1
            continue
            # print(f'\t{row[0]} is in the {row[1]} condition, and has disease {row[3]}.')
#            patient_record.append(single_patient)
    print(f'Processed {line_count} lines.')

val_record_5 = []

with open('lr_result_14AR-SGD-IBout_r0_n2.csv') as csv_file:
    #### alpha = 0.045

    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip the first four lines indicating the state of current program
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            col_name = row
            line_count += 1
        # read in line every 4 lines
        elif line_count%4 == 3:
            single_epoch = {}
            single_epoch['epoch'] = row[col_name.index('Epoch')]
            single_epoch['Loss'] = row[col_name.index('Loss')]
            single_epoch['Prec@1'] = row[col_name.index('Prec@1')]
            single_epoch['Prec@5'] = row[col_name.index('Prec@5')]
            line_count += 1
            val_record_5.append(single_epoch)
        else:
            line_count += 1
            continue
            # print(f'\t{row[0]} is in the {row[1]} condition, and has disease {row[3]}.')
#            patient_record.append(single_patient)
    print(f'Processed {line_count} lines.')

val_record_6 = []

with open('large_lr_result_6AR-SGD-IBout_r0_n2.csv') as csv_file:
    ### alpha = 0.08
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # skip the first four lines indicating the state of current program
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            col_name = row
            line_count += 1
        # read in line every 4 lines
        elif line_count%4 == 3:
            single_epoch = {}
            single_epoch['epoch'] = row[col_name.index('Epoch')]
            single_epoch['Loss'] = row[col_name.index('Loss')]
            single_epoch['Prec@1'] = row[col_name.index('Prec@1')]
            single_epoch['Prec@5'] = row[col_name.index('Prec@5')]
            line_count += 1
            val_record_6.append(single_epoch)
        else:
            line_count += 1
            continue
            # print(f'\t{row[0]} is in the {row[1]} condition, and has disease {row[3]}.')
#            patient_record.append(single_patient)
    print(f'Processed {line_count} lines.')

interest = 'Prec@1'

fig = plt.figure(figsize=(10, 10))
ax = plt.axes()  
x = [int(i['epoch']) for i in val_record_1]
y1 = [float(i[interest]) for i in val_record_1]
y2 = [float(i[interest]) for i in val_record_2]
y3 = [float(i[interest]) for i in val_record_3]
y4 = [float(i[interest]) for i in val_record_4[2:]]
#y5 = [float(i[interest]) for i in val_record_5]
y6 = [float(i[interest]) for i in val_record_6[2:]]


ax.plot(x, y1,'magenta',linewidth=4,label=r'$\alpha = 0.005$') 
ax.plot(x, y2,'blue',linewidth=4,label=r'$\alpha = 0.01$') 
ax.plot(x, y3,'black',linewidth=4,label=r'$\alpha = 0.02$') 
ax.plot(x, y4,'cyan',linewidth=4,label=r'$\alpha = 0.04$') 
# ax.plot(x, y5,'green',linewidth=4,label=r'$\alpha = 0.045$') 
ax.plot(x, y6,'red',linewidth=4,label=r'$\alpha = 0.08$') 


ax.set_xlabel('Epochs', fontsize = 22,fontweight='bold')
ax.set_ylabel(interest, fontsize = 22,fontweight='bold')
plt.yticks(fontsize = 16,fontweight='bold')
plt.xticks(fontsize = 16,fontweight='bold')

legend_properties = {'size':22,'weight':'bold'}
plt.legend(prop=legend_properties)
# plt.savefig('arsgd_loss.eps', format='eps')