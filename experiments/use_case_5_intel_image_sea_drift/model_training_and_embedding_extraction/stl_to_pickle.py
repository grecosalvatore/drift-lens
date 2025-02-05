import pickle
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np


def open_image(path):
    img = Image.open(path)  
    img_array = np.array(img)

    return img_array

def save_to(path, data):

    with open(path, 'wb') as file:
        data = pickle.dump(data, file)
    print(path)
    print("saved")

def pre_save(xx,yy):
    final=[]
    for i in range(len(xx)):
        final.append((xx[i], yy[i]))
    final=np.array(final)

    return final



data_path="Stl-10/"

data_tot=[]

drift=[]
label=0
for fold in os.listdir(data_path):
    try:
        print(fold)
        fold_path=os.path.join(data_path, fold)
        if fold!='truck':
            for i in os.listdir(fold_path):
                try:
                    img_path=os.path.join(fold_path, i)
                    img=open_image(img_path)
                    data_tot.append((img,label))
                except Exception as e1:
                    pass
        else:
            for i in os.listdir(fold_path):
                try:
                    img_path=os.path.join(fold_path, i)
                    img=open_image(img_path)
                    drift.append((img,label))
                except Exception as e2:
                    print(e2)
                    pass
    
        label=label+1
    
    except Exception as e3:
        pass


X=[]
Y=[]
for x,y in data_tot:
    X.append(x)
    Y.append(y)

X=np.array(X)
Y=np.array(Y)
drift=np.array(drift)


X_valtest, X_train, Y_valtest, y_train= train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=42)
X_test, X_val, y_test, y_val= train_test_split(X_valtest, Y_valtest, test_size=0.5, stratify=Y_valtest, random_state=42)

train_file_path = 'stl_train.pickle'
test_file_path = 'stl_test.pickle'
val_file_path = 'stl_val.pickle'
deg_file_path = 'stl_deg.pickle'

final_train=pre_save(X_train, y_train)
final_test=pre_save(X_test, y_test)
final_val=pre_save(X_val, y_val)

save_to(train_file_path, final_train)
save_to(test_file_path, final_test)
save_to(val_file_path, final_val)
save_to(deg_file_path, drift)

print(len(final_train))
print(len(final_test))
print(len(final_val))
print(len(drift))


    