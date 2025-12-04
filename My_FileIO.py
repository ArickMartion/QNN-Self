#!/usr/bin/env python
# coding: utf-8

# In[3]:


import io
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib
import xlwt
import numpy as np
import xlrd
import os

stdout,stde,stdi=sys.stdout,sys.stderr,sys.stdin
importlib.reload(sys)
#sys.setdefaultencoding("utf-8") #python默认为 str ，转为 utf-8
sys.stdout,sys.stderr,sys.stdein=stdout,stde,stdi


# In[4]:


"""1. Save list data as a local txt file"""
def txt_save(filename,data,if_print=False): # filename is the file path, data is the list of data to write
    file=open(filename,"a")
    for i in range(0,len(data)):
        s=str(data[i]).replace("[","").replace("]","") # Remove "[]"
        s=s.replace("'","").replace(",",'')+"\n"
        file.write(s)
    file.close()
    
    if if_print==True:
        print("保存成功！")
        
def txt_save_dict(path,data):
    with open(path,"w") as f:
        f.write("{\n")
        for key,value in data.items():
            f.write("'")
            if type(key) is not str:
                key=str(key)
            f.write(key)
            f.write("'")
            f.write(":")
            f.write(str(value))
            f.write("\n")
        f.write("\n}")


def excel_save(save_path,data,data_type="single",overlap_ExistFile=True):
    """
     data: a list or dictionary, where each element represents one column of data; 
     when saving, store it vertically
    """
    if overlap_ExistFile == False:
        if os.path.exists(save_path):
            print("The file already exists!")
            return 0
    
    # Create an Excel file
    book=xlwt.Workbook(encoding="utf-8",style_compression=0)
    # Create a sheet in the Excel file
    if data_type=="single":
        sheet=book.add_sheet("Sheet1",cell_overwrite_ok=True)

        for c in range(len(data)):
            for l in range(len(data[c])):
                sheet.write(l,c,data[c][l])
        book.save(save_path)
        print("# XLS format data saved successfully!")

    elif data_type=="multi":
        for key in data.keys():
            sheet=book.add_sheet(key,cell_overwrite_ok=True)
            for c in range(len(data[key])):
                for l in range(len(data[key][c])):
                    sheet.write(l,c,data[key][c][l])
            book.save(save_path)
        print("# XLS format data saved successfully!")
        


"""2. Define a class for reading Excel data"""
class Excel():
    def __init__(self):
        self.path=None
        #self.data=xlrd.open_workbook(self.path,"r")
        self.data=None
        self.sheet_names=None
        self.tabel=None
    
    def get_data(self,path=None):
        """Get sheet names and contents"""
        if path==None:
            path=self.path
        self.data=xlrd.open_workbook(path)
        self.sheet_names=self.data.sheet_names() 
        return self.sheet_names
    
    def get_table(self,sheet_name="Sheet1"):
        """Get sheet names"""
        self.tabel=self.data.sheet_by_name(sheet_name) 
        self.data.sheet_loaded(sheet_name) 
        return self.tabel
        
    def read_data(self,tabel,col=0,start_rowx=0,end_rowx=None):
        """Get data from column `col`, rows `start_rowx` to `end_rowx`"""
        if type(col)==str:
            if len(col)==1:
                idx=ord(col)-65
            elif len(col)==2:
                idx=26 
                idx+=(ord(col[0])-65)*26+ord(col[1])-65
        elif type(col)==int:
            idx=col
        data=tabel.col_values(idx,start_rowx=start_rowx,end_rowx=end_rowx) 
        return data
    
    def data_filter_None(self,x,y):
        """Handle and remove parts that are None"""
        new_x,new_y=[],[]
        for idx,k in enumerate(y):
            if k!="":
                new_x.append(x[idx])
                new_y.append(k)
        return np.array(new_x),np.array(new_y)


# In[6]:


"""3. Define a class for reading CSV files"""
class my_csv():
    def __init__(self):
        self.root_path = None
        self.file_path = None 

    def get_headerrow(self,file_path=None):
        """Get header row information"""
        if file_path==None:
            file_path=self.file_path
        with io.open(file_path,"r") as file:
            reader=csv.reader(file)
            row=next(reader)
            return row
        
    def read_csv(self,file_path=None):
        """Read all information including the header row"""
        if file_path is None:
            file_path=self.file_path
        with io.open(file_path,"r",encoding="utf-8") as file:
            reader=csv.reader(file)
            rows=[row for row in reader]
            return rows





