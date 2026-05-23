import numpy as np
arr_2d=np.array([[2,1,3],[4,6,5]])
reshaped=arr_2d.reshape(3,2)#reshapes the vector
flattened=arr_2d.flatten()#flattens the vector
transposed=arr_2d.T

#print(reshaped)
#print(arr_2d.shape)
#print(arr_2d.size)
#print(arr_2d.ndim)#gives number of dimension
#print(arr_2d.dtype)
#print(arr_2d)
arr_range=np.arange(100,1000,2)#creates vector with start 100 end 1000 and step a 2
zero_array=np.zeros((3,4))#creates a vector with all elements 0 and of size 3*4
#print(zero_array)
one_array=np.ones((5,4))
#print(one_array)
full=np.full((2,3),6)#creates vector of size 2*3 and with each value as 6
random_arr=np.random.random((2,3))#gives vector with size 2*3 and each value is random and between 0 &1 
arr=np.arange(10)
#print(arr[1:9:2])#standard slicing of array
#print(arr_2d[1,1])#gives element of second row and second column
#print(arr_2d[1])#gives entire second row
#print(arr_2d[:,1])#gives entire second column
unsorted=np.array([3,9,5,7,8,3,5])
#print(np.sort(arr_2d))#sorts the array
#print(np.sort(arr_2d,axis=0))#sorts along the columns
#print(np.sort(arr_2d,axis=1))#sorts along the rows
odd=unsorted[unsorted%2==1]#all elements satisfying the condition are entered into the odd array
#print(odd)
mask=unsorted>5#basically the condition is stored in the name mask
greater=unsorted[mask]
#print(greater)
indices=[0,3,5]
#print(unsorted[indices])#basically this is another way of passing indexes for arrays
whereuse=np.where(unsorted>3,unsorted,unsorted*2)#basically condition is cheked on each element and if condtion is true then second item replaces that element and if not than last
#print(whereuse)
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
arr3=np.concatenate((arr1,arr2))
#print(arr3)
new_row=np.array([7,8,9])
with_new_row=np.vstack((arr_2d,new_row))#adds new row
#print(with_new_row)
new_col=np.array([[7],[8]])
with_new_col=np.hstack((arr_2d,new_col))#adds new col
#print(with_new_col)
summed=np.sum(arr_2d[:,1:],axis=0)#basically sums the given array with given axis
#print(summed)
mini=np.min(arr_2d,axis=1)#finds the minimum value in each row
avg=np.mean(arr_2d,axis=1)#find the average of each row
cumilative=np.cumsum(arr_2d,axis=1)#gives the cumulative sum
#print(mini)
a=np.array([1,2,3])
b=np.array([4,5,-2])
dotpro=np.dot(a,b)#gives dotproduct
#print(dotpro)
angle=np.arccos(dotpro/(np.linalg.norm(a)*np.linalg.norm(b)))#norm gives magnitude of vectors and arccos give value in radian
#print(angle)
names=np.array(['harsh','tia','nigga'])
vectorized_upper=np.vectorize(str.upper)#this basically vectorises the operation that is it performs it with each element of the array
#print(vectorized_upper(names))#gives the array names with each cahracter upper in each word
c=np.array([1,2,3])
d=np.array([[5],[10],[15]])
matmul=c@d#for matrix multiplication
#print(matmul)

