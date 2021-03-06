??%
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8?? 
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
%res_block/cnn_block_3/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%res_block/cnn_block_3/conv2d_3/kernel
?
9res_block/cnn_block_3/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp%res_block/cnn_block_3/conv2d_3/kernel*&
_output_shapes
: *
dtype0
?
#res_block/cnn_block_3/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#res_block/cnn_block_3/conv2d_3/bias
?
7res_block/cnn_block_3/conv2d_3/bias/Read/ReadVariableOpReadVariableOp#res_block/cnn_block_3/conv2d_3/bias*
_output_shapes
: *
dtype0
?
1res_block/cnn_block_3/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31res_block/cnn_block_3/batch_normalization_3/gamma
?
Eres_block/cnn_block_3/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp1res_block/cnn_block_3/batch_normalization_3/gamma*
_output_shapes
: *
dtype0
?
0res_block/cnn_block_3/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20res_block/cnn_block_3/batch_normalization_3/beta
?
Dres_block/cnn_block_3/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp0res_block/cnn_block_3/batch_normalization_3/beta*
_output_shapes
: *
dtype0
?
%res_block/cnn_block_4/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *6
shared_name'%res_block/cnn_block_4/conv2d_4/kernel
?
9res_block/cnn_block_4/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp%res_block/cnn_block_4/conv2d_4/kernel*&
_output_shapes
:  *
dtype0
?
#res_block/cnn_block_4/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#res_block/cnn_block_4/conv2d_4/bias
?
7res_block/cnn_block_4/conv2d_4/bias/Read/ReadVariableOpReadVariableOp#res_block/cnn_block_4/conv2d_4/bias*
_output_shapes
: *
dtype0
?
1res_block/cnn_block_4/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31res_block/cnn_block_4/batch_normalization_4/gamma
?
Eres_block/cnn_block_4/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp1res_block/cnn_block_4/batch_normalization_4/gamma*
_output_shapes
: *
dtype0
?
0res_block/cnn_block_4/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20res_block/cnn_block_4/batch_normalization_4/beta
?
Dres_block/cnn_block_4/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp0res_block/cnn_block_4/batch_normalization_4/beta*
_output_shapes
: *
dtype0
?
%res_block/cnn_block_5/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%res_block/cnn_block_5/conv2d_5/kernel
?
9res_block/cnn_block_5/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp%res_block/cnn_block_5/conv2d_5/kernel*&
_output_shapes
: @*
dtype0
?
#res_block/cnn_block_5/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#res_block/cnn_block_5/conv2d_5/bias
?
7res_block/cnn_block_5/conv2d_5/bias/Read/ReadVariableOpReadVariableOp#res_block/cnn_block_5/conv2d_5/bias*
_output_shapes
:@*
dtype0
?
1res_block/cnn_block_5/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31res_block/cnn_block_5/batch_normalization_5/gamma
?
Eres_block/cnn_block_5/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp1res_block/cnn_block_5/batch_normalization_5/gamma*
_output_shapes
:@*
dtype0
?
0res_block/cnn_block_5/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20res_block/cnn_block_5/batch_normalization_5/beta
?
Dres_block/cnn_block_5/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp0res_block/cnn_block_5/batch_normalization_5/beta*
_output_shapes
:@*
dtype0
?
res_block/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameres_block/conv2d_6/kernel
?
-res_block/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpres_block/conv2d_6/kernel*&
_output_shapes
: *
dtype0
?
res_block/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameres_block/conv2d_6/bias

+res_block/conv2d_6/bias/Read/ReadVariableOpReadVariableOpres_block/conv2d_6/bias*
_output_shapes
: *
dtype0
?
'res_block_1/cnn_block_6/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*8
shared_name)'res_block_1/cnn_block_6/conv2d_7/kernel
?
;res_block_1/cnn_block_6/conv2d_7/kernel/Read/ReadVariableOpReadVariableOp'res_block_1/cnn_block_6/conv2d_7/kernel*'
_output_shapes
:@?*
dtype0
?
%res_block_1/cnn_block_6/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%res_block_1/cnn_block_6/conv2d_7/bias
?
9res_block_1/cnn_block_6/conv2d_7/bias/Read/ReadVariableOpReadVariableOp%res_block_1/cnn_block_6/conv2d_7/bias*
_output_shapes	
:?*
dtype0
?
3res_block_1/cnn_block_6/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53res_block_1/cnn_block_6/batch_normalization_6/gamma
?
Gres_block_1/cnn_block_6/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp3res_block_1/cnn_block_6/batch_normalization_6/gamma*
_output_shapes	
:?*
dtype0
?
2res_block_1/cnn_block_6/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*C
shared_name42res_block_1/cnn_block_6/batch_normalization_6/beta
?
Fres_block_1/cnn_block_6/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp2res_block_1/cnn_block_6/batch_normalization_6/beta*
_output_shapes	
:?*
dtype0
?
'res_block_1/cnn_block_7/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'res_block_1/cnn_block_7/conv2d_8/kernel
?
;res_block_1/cnn_block_7/conv2d_8/kernel/Read/ReadVariableOpReadVariableOp'res_block_1/cnn_block_7/conv2d_8/kernel*(
_output_shapes
:??*
dtype0
?
%res_block_1/cnn_block_7/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%res_block_1/cnn_block_7/conv2d_8/bias
?
9res_block_1/cnn_block_7/conv2d_8/bias/Read/ReadVariableOpReadVariableOp%res_block_1/cnn_block_7/conv2d_8/bias*
_output_shapes	
:?*
dtype0
?
3res_block_1/cnn_block_7/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53res_block_1/cnn_block_7/batch_normalization_7/gamma
?
Gres_block_1/cnn_block_7/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp3res_block_1/cnn_block_7/batch_normalization_7/gamma*
_output_shapes	
:?*
dtype0
?
2res_block_1/cnn_block_7/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*C
shared_name42res_block_1/cnn_block_7/batch_normalization_7/beta
?
Fres_block_1/cnn_block_7/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp2res_block_1/cnn_block_7/batch_normalization_7/beta*
_output_shapes	
:?*
dtype0
?
'res_block_1/cnn_block_8/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'res_block_1/cnn_block_8/conv2d_9/kernel
?
;res_block_1/cnn_block_8/conv2d_9/kernel/Read/ReadVariableOpReadVariableOp'res_block_1/cnn_block_8/conv2d_9/kernel*(
_output_shapes
:??*
dtype0
?
%res_block_1/cnn_block_8/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%res_block_1/cnn_block_8/conv2d_9/bias
?
9res_block_1/cnn_block_8/conv2d_9/bias/Read/ReadVariableOpReadVariableOp%res_block_1/cnn_block_8/conv2d_9/bias*
_output_shapes	
:?*
dtype0
?
3res_block_1/cnn_block_8/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53res_block_1/cnn_block_8/batch_normalization_8/gamma
?
Gres_block_1/cnn_block_8/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp3res_block_1/cnn_block_8/batch_normalization_8/gamma*
_output_shapes	
:?*
dtype0
?
2res_block_1/cnn_block_8/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*C
shared_name42res_block_1/cnn_block_8/batch_normalization_8/beta
?
Fres_block_1/cnn_block_8/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp2res_block_1/cnn_block_8/batch_normalization_8/beta*
_output_shapes	
:?*
dtype0
?
res_block_1/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*-
shared_nameres_block_1/conv2d_10/kernel
?
0res_block_1/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpres_block_1/conv2d_10/kernel*'
_output_shapes
:@?*
dtype0
?
res_block_1/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameres_block_1/conv2d_10/bias
?
.res_block_1/conv2d_10/bias/Read/ReadVariableOpReadVariableOpres_block_1/conv2d_10/bias*
_output_shapes	
:?*
dtype0
?
7res_block/cnn_block_3/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97res_block/cnn_block_3/batch_normalization_3/moving_mean
?
Kres_block/cnn_block_3/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp7res_block/cnn_block_3/batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
?
;res_block/cnn_block_3/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;res_block/cnn_block_3/batch_normalization_3/moving_variance
?
Ores_block/cnn_block_3/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp;res_block/cnn_block_3/batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
?
7res_block/cnn_block_4/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97res_block/cnn_block_4/batch_normalization_4/moving_mean
?
Kres_block/cnn_block_4/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp7res_block/cnn_block_4/batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
?
;res_block/cnn_block_4/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;res_block/cnn_block_4/batch_normalization_4/moving_variance
?
Ores_block/cnn_block_4/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp;res_block/cnn_block_4/batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
?
7res_block/cnn_block_5/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97res_block/cnn_block_5/batch_normalization_5/moving_mean
?
Kres_block/cnn_block_5/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp7res_block/cnn_block_5/batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
?
;res_block/cnn_block_5/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;res_block/cnn_block_5/batch_normalization_5/moving_variance
?
Ores_block/cnn_block_5/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp;res_block/cnn_block_5/batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
?
9res_block_1/cnn_block_6/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9res_block_1/cnn_block_6/batch_normalization_6/moving_mean
?
Mres_block_1/cnn_block_6/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp9res_block_1/cnn_block_6/batch_normalization_6/moving_mean*
_output_shapes	
:?*
dtype0
?
=res_block_1/cnn_block_6/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*N
shared_name?=res_block_1/cnn_block_6/batch_normalization_6/moving_variance
?
Qres_block_1/cnn_block_6/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp=res_block_1/cnn_block_6/batch_normalization_6/moving_variance*
_output_shapes	
:?*
dtype0
?
9res_block_1/cnn_block_7/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9res_block_1/cnn_block_7/batch_normalization_7/moving_mean
?
Mres_block_1/cnn_block_7/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp9res_block_1/cnn_block_7/batch_normalization_7/moving_mean*
_output_shapes	
:?*
dtype0
?
=res_block_1/cnn_block_7/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*N
shared_name?=res_block_1/cnn_block_7/batch_normalization_7/moving_variance
?
Qres_block_1/cnn_block_7/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp=res_block_1/cnn_block_7/batch_normalization_7/moving_variance*
_output_shapes	
:?*
dtype0
?
9res_block_1/cnn_block_8/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9res_block_1/cnn_block_8/batch_normalization_8/moving_mean
?
Mres_block_1/cnn_block_8/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp9res_block_1/cnn_block_8/batch_normalization_8/moving_mean*
_output_shapes	
:?*
dtype0
?
=res_block_1/cnn_block_8/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*N
shared_name?=res_block_1/cnn_block_8/batch_normalization_8/moving_variance
?
Qres_block_1/cnn_block_8/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp=res_block_1/cnn_block_8/batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
??
*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:
*
dtype0
?
,Adam/res_block/cnn_block_3/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/res_block/cnn_block_3/conv2d_3/kernel/m
?
@Adam/res_block/cnn_block_3/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_3/conv2d_3/kernel/m*&
_output_shapes
: *
dtype0
?
*Adam/res_block/cnn_block_3/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/res_block/cnn_block_3/conv2d_3/bias/m
?
>Adam/res_block/cnn_block_3/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_3/conv2d_3/bias/m*
_output_shapes
: *
dtype0
?
8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/m
?
LAdam/res_block/cnn_block_3/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/m*
_output_shapes
: *
dtype0
?
7Adam/res_block/cnn_block_3/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/res_block/cnn_block_3/batch_normalization_3/beta/m
?
KAdam/res_block/cnn_block_3/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_3/batch_normalization_3/beta/m*
_output_shapes
: *
dtype0
?
,Adam/res_block/cnn_block_4/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *=
shared_name.,Adam/res_block/cnn_block_4/conv2d_4/kernel/m
?
@Adam/res_block/cnn_block_4/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_4/conv2d_4/kernel/m*&
_output_shapes
:  *
dtype0
?
*Adam/res_block/cnn_block_4/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/res_block/cnn_block_4/conv2d_4/bias/m
?
>Adam/res_block/cnn_block_4/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_4/conv2d_4/bias/m*
_output_shapes
: *
dtype0
?
8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/m
?
LAdam/res_block/cnn_block_4/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/m*
_output_shapes
: *
dtype0
?
7Adam/res_block/cnn_block_4/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/res_block/cnn_block_4/batch_normalization_4/beta/m
?
KAdam/res_block/cnn_block_4/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_4/batch_normalization_4/beta/m*
_output_shapes
: *
dtype0
?
,Adam/res_block/cnn_block_5/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/res_block/cnn_block_5/conv2d_5/kernel/m
?
@Adam/res_block/cnn_block_5/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_5/conv2d_5/kernel/m*&
_output_shapes
: @*
dtype0
?
*Adam/res_block/cnn_block_5/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/res_block/cnn_block_5/conv2d_5/bias/m
?
>Adam/res_block/cnn_block_5/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_5/conv2d_5/bias/m*
_output_shapes
:@*
dtype0
?
8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/m
?
LAdam/res_block/cnn_block_5/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/m*
_output_shapes
:@*
dtype0
?
7Adam/res_block/cnn_block_5/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/res_block/cnn_block_5/batch_normalization_5/beta/m
?
KAdam/res_block/cnn_block_5/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_5/batch_normalization_5/beta/m*
_output_shapes
:@*
dtype0
?
 Adam/res_block/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/res_block/conv2d_6/kernel/m
?
4Adam/res_block/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/res_block/conv2d_6/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/res_block/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/res_block/conv2d_6/bias/m
?
2Adam/res_block/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/res_block/conv2d_6/bias/m*
_output_shapes
: *
dtype0
?
.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*?
shared_name0.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/m
?
BAdam/res_block_1/cnn_block_6/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/m*'
_output_shapes
:@?*
dtype0
?
,Adam/res_block_1/cnn_block_6/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/res_block_1/cnn_block_6/conv2d_7/bias/m
?
@Adam/res_block_1/cnn_block_6/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_6/conv2d_7/bias/m*
_output_shapes	
:?*
dtype0
?
:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*K
shared_name<:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m
?
NAdam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m*
_output_shapes	
:?*
dtype0
?
9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/m
?
MAdam/res_block_1/cnn_block_6/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*?
shared_name0.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/m
?
BAdam/res_block_1/cnn_block_7/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/m*(
_output_shapes
:??*
dtype0
?
,Adam/res_block_1/cnn_block_7/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/res_block_1/cnn_block_7/conv2d_8/bias/m
?
@Adam/res_block_1/cnn_block_7/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_7/conv2d_8/bias/m*
_output_shapes	
:?*
dtype0
?
:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*K
shared_name<:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m
?
NAdam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m*
_output_shapes	
:?*
dtype0
?
9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/m
?
MAdam/res_block_1/cnn_block_7/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*?
shared_name0.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/m
?
BAdam/res_block_1/cnn_block_8/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/m*(
_output_shapes
:??*
dtype0
?
,Adam/res_block_1/cnn_block_8/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/res_block_1/cnn_block_8/conv2d_9/bias/m
?
@Adam/res_block_1/cnn_block_8/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_8/conv2d_9/bias/m*
_output_shapes	
:?*
dtype0
?
:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*K
shared_name<:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m
?
NAdam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m*
_output_shapes	
:?*
dtype0
?
9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/m
?
MAdam/res_block_1/cnn_block_8/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/res_block_1/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*4
shared_name%#Adam/res_block_1/conv2d_10/kernel/m
?
7Adam/res_block_1/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/res_block_1/conv2d_10/kernel/m*'
_output_shapes
:@?*
dtype0
?
!Adam/res_block_1/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/res_block_1/conv2d_10/bias/m
?
5Adam/res_block_1/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOp!Adam/res_block_1/conv2d_10/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??
*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
??
*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:
*
dtype0
?
,Adam/res_block/cnn_block_3/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/res_block/cnn_block_3/conv2d_3/kernel/v
?
@Adam/res_block/cnn_block_3/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_3/conv2d_3/kernel/v*&
_output_shapes
: *
dtype0
?
*Adam/res_block/cnn_block_3/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/res_block/cnn_block_3/conv2d_3/bias/v
?
>Adam/res_block/cnn_block_3/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_3/conv2d_3/bias/v*
_output_shapes
: *
dtype0
?
8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/v
?
LAdam/res_block/cnn_block_3/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/v*
_output_shapes
: *
dtype0
?
7Adam/res_block/cnn_block_3/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/res_block/cnn_block_3/batch_normalization_3/beta/v
?
KAdam/res_block/cnn_block_3/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_3/batch_normalization_3/beta/v*
_output_shapes
: *
dtype0
?
,Adam/res_block/cnn_block_4/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *=
shared_name.,Adam/res_block/cnn_block_4/conv2d_4/kernel/v
?
@Adam/res_block/cnn_block_4/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_4/conv2d_4/kernel/v*&
_output_shapes
:  *
dtype0
?
*Adam/res_block/cnn_block_4/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/res_block/cnn_block_4/conv2d_4/bias/v
?
>Adam/res_block/cnn_block_4/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_4/conv2d_4/bias/v*
_output_shapes
: *
dtype0
?
8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/v
?
LAdam/res_block/cnn_block_4/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/v*
_output_shapes
: *
dtype0
?
7Adam/res_block/cnn_block_4/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/res_block/cnn_block_4/batch_normalization_4/beta/v
?
KAdam/res_block/cnn_block_4/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_4/batch_normalization_4/beta/v*
_output_shapes
: *
dtype0
?
,Adam/res_block/cnn_block_5/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/res_block/cnn_block_5/conv2d_5/kernel/v
?
@Adam/res_block/cnn_block_5/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/res_block/cnn_block_5/conv2d_5/kernel/v*&
_output_shapes
: @*
dtype0
?
*Adam/res_block/cnn_block_5/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/res_block/cnn_block_5/conv2d_5/bias/v
?
>Adam/res_block/cnn_block_5/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOp*Adam/res_block/cnn_block_5/conv2d_5/bias/v*
_output_shapes
:@*
dtype0
?
8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/v
?
LAdam/res_block/cnn_block_5/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/v*
_output_shapes
:@*
dtype0
?
7Adam/res_block/cnn_block_5/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/res_block/cnn_block_5/batch_normalization_5/beta/v
?
KAdam/res_block/cnn_block_5/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp7Adam/res_block/cnn_block_5/batch_normalization_5/beta/v*
_output_shapes
:@*
dtype0
?
 Adam/res_block/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/res_block/conv2d_6/kernel/v
?
4Adam/res_block/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/res_block/conv2d_6/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/res_block/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/res_block/conv2d_6/bias/v
?
2Adam/res_block/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/res_block/conv2d_6/bias/v*
_output_shapes
: *
dtype0
?
.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*?
shared_name0.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/v
?
BAdam/res_block_1/cnn_block_6/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/v*'
_output_shapes
:@?*
dtype0
?
,Adam/res_block_1/cnn_block_6/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/res_block_1/cnn_block_6/conv2d_7/bias/v
?
@Adam/res_block_1/cnn_block_6/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_6/conv2d_7/bias/v*
_output_shapes	
:?*
dtype0
?
:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*K
shared_name<:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v
?
NAdam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v*
_output_shapes	
:?*
dtype0
?
9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/v
?
MAdam/res_block_1/cnn_block_6/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*?
shared_name0.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/v
?
BAdam/res_block_1/cnn_block_7/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/v*(
_output_shapes
:??*
dtype0
?
,Adam/res_block_1/cnn_block_7/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/res_block_1/cnn_block_7/conv2d_8/bias/v
?
@Adam/res_block_1/cnn_block_7/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_7/conv2d_8/bias/v*
_output_shapes	
:?*
dtype0
?
:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*K
shared_name<:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v
?
NAdam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v*
_output_shapes	
:?*
dtype0
?
9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/v
?
MAdam/res_block_1/cnn_block_7/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*?
shared_name0.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/v
?
BAdam/res_block_1/cnn_block_8/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/v*(
_output_shapes
:??*
dtype0
?
,Adam/res_block_1/cnn_block_8/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/res_block_1/cnn_block_8/conv2d_9/bias/v
?
@Adam/res_block_1/cnn_block_8/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOp,Adam/res_block_1/cnn_block_8/conv2d_9/bias/v*
_output_shapes	
:?*
dtype0
?
:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*K
shared_name<:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v
?
NAdam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v*
_output_shapes	
:?*
dtype0
?
9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*J
shared_name;9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/v
?
MAdam/res_block_1/cnn_block_8/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/res_block_1/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*4
shared_name%#Adam/res_block_1/conv2d_10/kernel/v
?
7Adam/res_block_1/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/res_block_1/conv2d_10/kernel/v*'
_output_shapes
:@?*
dtype0
?
!Adam/res_block_1/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/res_block_1/conv2d_10/bias/v
?
5Adam/res_block_1/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOp!Adam/res_block_1/conv2d_10/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
 
?
channels
cnn1
cnn2
cnn3
pooling
identity_mapping
trainable_variables
	variables
regularization_losses
	keras_api
?
channels
cnn1
cnn2
cnn3
pooling
identity_mapping
trainable_variables
	variables
regularization_losses
	keras_api
R
 trainable_variables
!	variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
?
*iter

+beta_1

,beta_2
	-decay
.learning_rate$m?%m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?$v?%v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
=14
>15
?16
@17
A18
B19
C20
D21
E22
F23
G24
H25
I26
J27
$28
%29
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
K14
L15
M16
N17
O18
P19
=20
>21
?22
@23
A24
B25
C26
D27
E28
F29
G30
H31
I32
J33
Q34
R35
S36
T37
U38
V39
$40
%41
 
?
trainable_variables
Wmetrics
Xnon_trainable_variables
	variables
Ylayer_metrics

Zlayers
	regularization_losses
[layer_regularization_losses
 
 
d
\conv
]bn
^trainable_variables
_	variables
`regularization_losses
a	keras_api
d
bconv
cbn
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
d
hconv
ibn
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
R
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
h

;kernel
<bias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
f
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
K14
L15
M16
N17
O18
P19
 
?
trainable_variables
vmetrics
wnon_trainable_variables
xlayer_metrics
	variables

ylayers
regularization_losses
zlayer_regularization_losses
 
e
{conv
|bn
}trainable_variables
~	variables
regularization_losses
?	keras_api
j
	?conv
?bn
?trainable_variables
?	variables
?regularization_losses
?	keras_api
j
	?conv
?bn
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Ikernel
Jbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
f
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
?
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
Q14
R15
S16
T17
U18
V19
 
?
trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
	variables
?layers
regularization_losses
 ?layer_regularization_losses
 
 
 
?
 trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
!	variables
?layers
"regularization_losses
 ?layer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
&trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
'	variables
?layers
(regularization_losses
 ?layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%res_block/cnn_block_3/conv2d_3/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#res_block/cnn_block_3/conv2d_3/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1res_block/cnn_block_3/batch_normalization_3/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0res_block/cnn_block_3/batch_normalization_3/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%res_block/cnn_block_4/conv2d_4/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#res_block/cnn_block_4/conv2d_4/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1res_block/cnn_block_4/batch_normalization_4/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0res_block/cnn_block_4/batch_normalization_4/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%res_block/cnn_block_5/conv2d_5/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#res_block/cnn_block_5/conv2d_5/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1res_block/cnn_block_5/batch_normalization_5/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0res_block/cnn_block_5/batch_normalization_5/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEres_block/conv2d_6/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEres_block/conv2d_6/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'res_block_1/cnn_block_6/conv2d_7/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%res_block_1/cnn_block_6/conv2d_7/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3res_block_1/cnn_block_6/batch_normalization_6/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2res_block_1/cnn_block_6/batch_normalization_6/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'res_block_1/cnn_block_7/conv2d_8/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%res_block_1/cnn_block_7/conv2d_8/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3res_block_1/cnn_block_7/batch_normalization_7/gamma1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2res_block_1/cnn_block_7/batch_normalization_7/beta1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'res_block_1/cnn_block_8/conv2d_9/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%res_block_1/cnn_block_8/conv2d_9/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3res_block_1/cnn_block_8/batch_normalization_8/gamma1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2res_block_1/cnn_block_8/batch_normalization_8/beta1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEres_block_1/conv2d_10/kernel1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEres_block_1/conv2d_10/bias1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7res_block/cnn_block_3/batch_normalization_3/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE;res_block/cnn_block_3/batch_normalization_3/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7res_block/cnn_block_4/batch_normalization_4/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE;res_block/cnn_block_4/batch_normalization_4/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7res_block/cnn_block_5/batch_normalization_5/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE;res_block/cnn_block_5/batch_normalization_5/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9res_block_1/cnn_block_6/batch_normalization_6/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=res_block_1/cnn_block_6/batch_normalization_6/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9res_block_1/cnn_block_7/batch_normalization_7/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=res_block_1/cnn_block_7/batch_normalization_7/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9res_block_1/cnn_block_8/batch_normalization_8/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=res_block_1/cnn_block_8/batch_normalization_8/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
V
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
 
#
0
1
2
3
4
 
l

/kernel
0bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	1gamma
2beta
Kmoving_mean
Lmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api

/0
01
12
23
*
/0
01
12
23
K4
L5
 
?
^trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
_	variables
?layers
`regularization_losses
 ?layer_regularization_losses
l

3kernel
4bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	5gamma
6beta
Mmoving_mean
Nmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api

30
41
52
63
*
30
41
52
63
M4
N5
 
?
dtrainable_variables
?metrics
?non_trainable_variables
?layer_metrics
e	variables
?layers
fregularization_losses
 ?layer_regularization_losses
l

7kernel
8bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	9gamma
:beta
Omoving_mean
Pmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api

70
81
92
:3
*
70
81
92
:3
O4
P5
 
?
jtrainable_variables
?metrics
?non_trainable_variables
?layer_metrics
k	variables
?layers
lregularization_losses
 ?layer_regularization_losses
 
 
 
?
ntrainable_variables
?metrics
?non_trainable_variables
?layer_metrics
o	variables
?layers
pregularization_losses
 ?layer_regularization_losses

;0
<1

;0
<1
 
?
rtrainable_variables
?metrics
?non_trainable_variables
?layer_metrics
s	variables
?layers
tregularization_losses
 ?layer_regularization_losses
 
*
K0
L1
M2
N3
O4
P5
 
#
0
1
2
3
4
 
l

=kernel
>bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	?gamma
@beta
Qmoving_mean
Rmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api

=0
>1
?2
@3
*
=0
>1
?2
@3
Q4
R5
 
?
}trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
~	variables
?layers
regularization_losses
 ?layer_regularization_losses
l

Akernel
Bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	Cgamma
Dbeta
Smoving_mean
Tmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api

A0
B1
C2
D3
*
A0
B1
C2
D3
S4
T5
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
l

Ekernel
Fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis
	Ggamma
Hbeta
Umoving_mean
Vmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api

E0
F1
G2
H3
*
E0
F1
G2
H3
U4
V5
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 
 
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses

I0
J1

I0
J1
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 
*
Q0
R1
S2
T3
U4
V5
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api

/0
01

/0
01
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

10
21

10
21
K2
L3
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

K0
L1
 

\0
]1
 

30
41

30
41
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

50
61

50
61
M2
N3
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

M0
N1
 

b0
c1
 

70
81

70
81
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

90
:1

90
:1
O2
P3
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

O0
P1
 

h0
i1
 
 
 
 
 
 
 
 
 
 
 

=0
>1

=0
>1
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

?0
@1

?0
@1
Q2
R3
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

Q0
R1
 

{0
|1
 

A0
B1

A0
B1
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

C0
D1

C0
D1
S2
T3
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

S0
T1
 

?0
?1
 

E0
F1

E0
F1
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

G0
H1

G0
H1
U2
V3
 
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
 

U0
V1
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
 

K0
L1
 
 
 
 
 
 
 
 
 

M0
N1
 
 
 
 
 
 
 
 
 

O0
P1
 
 
 
 
 
 
 
 
 

Q0
R1
 
 
 
 
 
 
 
 
 

S0
T1
 
 
 
 
 
 
 
 
 

U0
V1
 
 
 
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block/cnn_block_3/conv2d_3/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/res_block/cnn_block_3/conv2d_3/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/res_block/cnn_block_3/batch_normalization_3/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block/cnn_block_4/conv2d_4/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/res_block/cnn_block_4/conv2d_4/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/res_block/cnn_block_4/batch_normalization_4/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block/cnn_block_5/conv2d_5/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/res_block/cnn_block_5/conv2d_5/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/res_block/cnn_block_5/batch_normalization_5/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/res_block/conv2d_6/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/res_block/conv2d_6/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block_1/cnn_block_6/conv2d_7/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block_1/cnn_block_7/conv2d_8/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block_1/cnn_block_8/conv2d_9/bias/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/mMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/mMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/res_block_1/conv2d_10/kernel/mMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/res_block_1/conv2d_10/bias/mMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block/cnn_block_3/conv2d_3/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/res_block/cnn_block_3/conv2d_3/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/res_block/cnn_block_3/batch_normalization_3/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block/cnn_block_4/conv2d_4/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/res_block/cnn_block_4/conv2d_4/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/res_block/cnn_block_4/batch_normalization_4/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block/cnn_block_5/conv2d_5/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/res_block/cnn_block_5/conv2d_5/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/res_block/cnn_block_5/batch_normalization_5/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/res_block/conv2d_6/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/res_block/conv2d_6/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block_1/cnn_block_6/conv2d_7/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block_1/cnn_block_7/conv2d_8/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/res_block_1/cnn_block_8/conv2d_9/bias/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/vMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/vMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/res_block_1/conv2d_10/kernel/vMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/res_block_1/conv2d_10/bias/vMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1%res_block/cnn_block_3/conv2d_3/kernel#res_block/cnn_block_3/conv2d_3/bias1res_block/cnn_block_3/batch_normalization_3/gamma0res_block/cnn_block_3/batch_normalization_3/beta7res_block/cnn_block_3/batch_normalization_3/moving_mean;res_block/cnn_block_3/batch_normalization_3/moving_variance%res_block/cnn_block_4/conv2d_4/kernel#res_block/cnn_block_4/conv2d_4/bias1res_block/cnn_block_4/batch_normalization_4/gamma0res_block/cnn_block_4/batch_normalization_4/beta7res_block/cnn_block_4/batch_normalization_4/moving_mean;res_block/cnn_block_4/batch_normalization_4/moving_varianceres_block/conv2d_6/kernelres_block/conv2d_6/bias%res_block/cnn_block_5/conv2d_5/kernel#res_block/cnn_block_5/conv2d_5/bias1res_block/cnn_block_5/batch_normalization_5/gamma0res_block/cnn_block_5/batch_normalization_5/beta7res_block/cnn_block_5/batch_normalization_5/moving_mean;res_block/cnn_block_5/batch_normalization_5/moving_variance'res_block_1/cnn_block_6/conv2d_7/kernel%res_block_1/cnn_block_6/conv2d_7/bias3res_block_1/cnn_block_6/batch_normalization_6/gamma2res_block_1/cnn_block_6/batch_normalization_6/beta9res_block_1/cnn_block_6/batch_normalization_6/moving_mean=res_block_1/cnn_block_6/batch_normalization_6/moving_variance'res_block_1/cnn_block_7/conv2d_8/kernel%res_block_1/cnn_block_7/conv2d_8/bias3res_block_1/cnn_block_7/batch_normalization_7/gamma2res_block_1/cnn_block_7/batch_normalization_7/beta9res_block_1/cnn_block_7/batch_normalization_7/moving_mean=res_block_1/cnn_block_7/batch_normalization_7/moving_varianceres_block_1/conv2d_10/kernelres_block_1/conv2d_10/bias'res_block_1/cnn_block_8/conv2d_9/kernel%res_block_1/cnn_block_8/conv2d_9/bias3res_block_1/cnn_block_8/batch_normalization_8/gamma2res_block_1/cnn_block_8/batch_normalization_8/beta9res_block_1/cnn_block_8/batch_normalization_8/moving_mean=res_block_1/cnn_block_8/batch_normalization_8/moving_variancedense_2/kerneldense_2/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_7199
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?:
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9res_block/cnn_block_3/conv2d_3/kernel/Read/ReadVariableOp7res_block/cnn_block_3/conv2d_3/bias/Read/ReadVariableOpEres_block/cnn_block_3/batch_normalization_3/gamma/Read/ReadVariableOpDres_block/cnn_block_3/batch_normalization_3/beta/Read/ReadVariableOp9res_block/cnn_block_4/conv2d_4/kernel/Read/ReadVariableOp7res_block/cnn_block_4/conv2d_4/bias/Read/ReadVariableOpEres_block/cnn_block_4/batch_normalization_4/gamma/Read/ReadVariableOpDres_block/cnn_block_4/batch_normalization_4/beta/Read/ReadVariableOp9res_block/cnn_block_5/conv2d_5/kernel/Read/ReadVariableOp7res_block/cnn_block_5/conv2d_5/bias/Read/ReadVariableOpEres_block/cnn_block_5/batch_normalization_5/gamma/Read/ReadVariableOpDres_block/cnn_block_5/batch_normalization_5/beta/Read/ReadVariableOp-res_block/conv2d_6/kernel/Read/ReadVariableOp+res_block/conv2d_6/bias/Read/ReadVariableOp;res_block_1/cnn_block_6/conv2d_7/kernel/Read/ReadVariableOp9res_block_1/cnn_block_6/conv2d_7/bias/Read/ReadVariableOpGres_block_1/cnn_block_6/batch_normalization_6/gamma/Read/ReadVariableOpFres_block_1/cnn_block_6/batch_normalization_6/beta/Read/ReadVariableOp;res_block_1/cnn_block_7/conv2d_8/kernel/Read/ReadVariableOp9res_block_1/cnn_block_7/conv2d_8/bias/Read/ReadVariableOpGres_block_1/cnn_block_7/batch_normalization_7/gamma/Read/ReadVariableOpFres_block_1/cnn_block_7/batch_normalization_7/beta/Read/ReadVariableOp;res_block_1/cnn_block_8/conv2d_9/kernel/Read/ReadVariableOp9res_block_1/cnn_block_8/conv2d_9/bias/Read/ReadVariableOpGres_block_1/cnn_block_8/batch_normalization_8/gamma/Read/ReadVariableOpFres_block_1/cnn_block_8/batch_normalization_8/beta/Read/ReadVariableOp0res_block_1/conv2d_10/kernel/Read/ReadVariableOp.res_block_1/conv2d_10/bias/Read/ReadVariableOpKres_block/cnn_block_3/batch_normalization_3/moving_mean/Read/ReadVariableOpOres_block/cnn_block_3/batch_normalization_3/moving_variance/Read/ReadVariableOpKres_block/cnn_block_4/batch_normalization_4/moving_mean/Read/ReadVariableOpOres_block/cnn_block_4/batch_normalization_4/moving_variance/Read/ReadVariableOpKres_block/cnn_block_5/batch_normalization_5/moving_mean/Read/ReadVariableOpOres_block/cnn_block_5/batch_normalization_5/moving_variance/Read/ReadVariableOpMres_block_1/cnn_block_6/batch_normalization_6/moving_mean/Read/ReadVariableOpQres_block_1/cnn_block_6/batch_normalization_6/moving_variance/Read/ReadVariableOpMres_block_1/cnn_block_7/batch_normalization_7/moving_mean/Read/ReadVariableOpQres_block_1/cnn_block_7/batch_normalization_7/moving_variance/Read/ReadVariableOpMres_block_1/cnn_block_8/batch_normalization_8/moving_mean/Read/ReadVariableOpQres_block_1/cnn_block_8/batch_normalization_8/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp@Adam/res_block/cnn_block_3/conv2d_3/kernel/m/Read/ReadVariableOp>Adam/res_block/cnn_block_3/conv2d_3/bias/m/Read/ReadVariableOpLAdam/res_block/cnn_block_3/batch_normalization_3/gamma/m/Read/ReadVariableOpKAdam/res_block/cnn_block_3/batch_normalization_3/beta/m/Read/ReadVariableOp@Adam/res_block/cnn_block_4/conv2d_4/kernel/m/Read/ReadVariableOp>Adam/res_block/cnn_block_4/conv2d_4/bias/m/Read/ReadVariableOpLAdam/res_block/cnn_block_4/batch_normalization_4/gamma/m/Read/ReadVariableOpKAdam/res_block/cnn_block_4/batch_normalization_4/beta/m/Read/ReadVariableOp@Adam/res_block/cnn_block_5/conv2d_5/kernel/m/Read/ReadVariableOp>Adam/res_block/cnn_block_5/conv2d_5/bias/m/Read/ReadVariableOpLAdam/res_block/cnn_block_5/batch_normalization_5/gamma/m/Read/ReadVariableOpKAdam/res_block/cnn_block_5/batch_normalization_5/beta/m/Read/ReadVariableOp4Adam/res_block/conv2d_6/kernel/m/Read/ReadVariableOp2Adam/res_block/conv2d_6/bias/m/Read/ReadVariableOpBAdam/res_block_1/cnn_block_6/conv2d_7/kernel/m/Read/ReadVariableOp@Adam/res_block_1/cnn_block_6/conv2d_7/bias/m/Read/ReadVariableOpNAdam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m/Read/ReadVariableOpMAdam/res_block_1/cnn_block_6/batch_normalization_6/beta/m/Read/ReadVariableOpBAdam/res_block_1/cnn_block_7/conv2d_8/kernel/m/Read/ReadVariableOp@Adam/res_block_1/cnn_block_7/conv2d_8/bias/m/Read/ReadVariableOpNAdam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m/Read/ReadVariableOpMAdam/res_block_1/cnn_block_7/batch_normalization_7/beta/m/Read/ReadVariableOpBAdam/res_block_1/cnn_block_8/conv2d_9/kernel/m/Read/ReadVariableOp@Adam/res_block_1/cnn_block_8/conv2d_9/bias/m/Read/ReadVariableOpNAdam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m/Read/ReadVariableOpMAdam/res_block_1/cnn_block_8/batch_normalization_8/beta/m/Read/ReadVariableOp7Adam/res_block_1/conv2d_10/kernel/m/Read/ReadVariableOp5Adam/res_block_1/conv2d_10/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp@Adam/res_block/cnn_block_3/conv2d_3/kernel/v/Read/ReadVariableOp>Adam/res_block/cnn_block_3/conv2d_3/bias/v/Read/ReadVariableOpLAdam/res_block/cnn_block_3/batch_normalization_3/gamma/v/Read/ReadVariableOpKAdam/res_block/cnn_block_3/batch_normalization_3/beta/v/Read/ReadVariableOp@Adam/res_block/cnn_block_4/conv2d_4/kernel/v/Read/ReadVariableOp>Adam/res_block/cnn_block_4/conv2d_4/bias/v/Read/ReadVariableOpLAdam/res_block/cnn_block_4/batch_normalization_4/gamma/v/Read/ReadVariableOpKAdam/res_block/cnn_block_4/batch_normalization_4/beta/v/Read/ReadVariableOp@Adam/res_block/cnn_block_5/conv2d_5/kernel/v/Read/ReadVariableOp>Adam/res_block/cnn_block_5/conv2d_5/bias/v/Read/ReadVariableOpLAdam/res_block/cnn_block_5/batch_normalization_5/gamma/v/Read/ReadVariableOpKAdam/res_block/cnn_block_5/batch_normalization_5/beta/v/Read/ReadVariableOp4Adam/res_block/conv2d_6/kernel/v/Read/ReadVariableOp2Adam/res_block/conv2d_6/bias/v/Read/ReadVariableOpBAdam/res_block_1/cnn_block_6/conv2d_7/kernel/v/Read/ReadVariableOp@Adam/res_block_1/cnn_block_6/conv2d_7/bias/v/Read/ReadVariableOpNAdam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v/Read/ReadVariableOpMAdam/res_block_1/cnn_block_6/batch_normalization_6/beta/v/Read/ReadVariableOpBAdam/res_block_1/cnn_block_7/conv2d_8/kernel/v/Read/ReadVariableOp@Adam/res_block_1/cnn_block_7/conv2d_8/bias/v/Read/ReadVariableOpNAdam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v/Read/ReadVariableOpMAdam/res_block_1/cnn_block_7/batch_normalization_7/beta/v/Read/ReadVariableOpBAdam/res_block_1/cnn_block_8/conv2d_9/kernel/v/Read/ReadVariableOp@Adam/res_block_1/cnn_block_8/conv2d_9/bias/v/Read/ReadVariableOpNAdam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v/Read/ReadVariableOpMAdam/res_block_1/cnn_block_8/batch_normalization_8/beta/v/Read/ReadVariableOp7Adam/res_block_1/conv2d_10/kernel/v/Read/ReadVariableOp5Adam/res_block_1/conv2d_10/bias/v/Read/ReadVariableOpConst*|
Tinu
s2q	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_8947
?)
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%res_block/cnn_block_3/conv2d_3/kernel#res_block/cnn_block_3/conv2d_3/bias1res_block/cnn_block_3/batch_normalization_3/gamma0res_block/cnn_block_3/batch_normalization_3/beta%res_block/cnn_block_4/conv2d_4/kernel#res_block/cnn_block_4/conv2d_4/bias1res_block/cnn_block_4/batch_normalization_4/gamma0res_block/cnn_block_4/batch_normalization_4/beta%res_block/cnn_block_5/conv2d_5/kernel#res_block/cnn_block_5/conv2d_5/bias1res_block/cnn_block_5/batch_normalization_5/gamma0res_block/cnn_block_5/batch_normalization_5/betares_block/conv2d_6/kernelres_block/conv2d_6/bias'res_block_1/cnn_block_6/conv2d_7/kernel%res_block_1/cnn_block_6/conv2d_7/bias3res_block_1/cnn_block_6/batch_normalization_6/gamma2res_block_1/cnn_block_6/batch_normalization_6/beta'res_block_1/cnn_block_7/conv2d_8/kernel%res_block_1/cnn_block_7/conv2d_8/bias3res_block_1/cnn_block_7/batch_normalization_7/gamma2res_block_1/cnn_block_7/batch_normalization_7/beta'res_block_1/cnn_block_8/conv2d_9/kernel%res_block_1/cnn_block_8/conv2d_9/bias3res_block_1/cnn_block_8/batch_normalization_8/gamma2res_block_1/cnn_block_8/batch_normalization_8/betares_block_1/conv2d_10/kernelres_block_1/conv2d_10/bias7res_block/cnn_block_3/batch_normalization_3/moving_mean;res_block/cnn_block_3/batch_normalization_3/moving_variance7res_block/cnn_block_4/batch_normalization_4/moving_mean;res_block/cnn_block_4/batch_normalization_4/moving_variance7res_block/cnn_block_5/batch_normalization_5/moving_mean;res_block/cnn_block_5/batch_normalization_5/moving_variance9res_block_1/cnn_block_6/batch_normalization_6/moving_mean=res_block_1/cnn_block_6/batch_normalization_6/moving_variance9res_block_1/cnn_block_7/batch_normalization_7/moving_mean=res_block_1/cnn_block_7/batch_normalization_7/moving_variance9res_block_1/cnn_block_8/batch_normalization_8/moving_mean=res_block_1/cnn_block_8/batch_normalization_8/moving_variancetotalcounttotal_1count_1Adam/dense_2/kernel/mAdam/dense_2/bias/m,Adam/res_block/cnn_block_3/conv2d_3/kernel/m*Adam/res_block/cnn_block_3/conv2d_3/bias/m8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/m7Adam/res_block/cnn_block_3/batch_normalization_3/beta/m,Adam/res_block/cnn_block_4/conv2d_4/kernel/m*Adam/res_block/cnn_block_4/conv2d_4/bias/m8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/m7Adam/res_block/cnn_block_4/batch_normalization_4/beta/m,Adam/res_block/cnn_block_5/conv2d_5/kernel/m*Adam/res_block/cnn_block_5/conv2d_5/bias/m8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/m7Adam/res_block/cnn_block_5/batch_normalization_5/beta/m Adam/res_block/conv2d_6/kernel/mAdam/res_block/conv2d_6/bias/m.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/m,Adam/res_block_1/cnn_block_6/conv2d_7/bias/m:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/m.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/m,Adam/res_block_1/cnn_block_7/conv2d_8/bias/m:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/m.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/m,Adam/res_block_1/cnn_block_8/conv2d_9/bias/m:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/m#Adam/res_block_1/conv2d_10/kernel/m!Adam/res_block_1/conv2d_10/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/v,Adam/res_block/cnn_block_3/conv2d_3/kernel/v*Adam/res_block/cnn_block_3/conv2d_3/bias/v8Adam/res_block/cnn_block_3/batch_normalization_3/gamma/v7Adam/res_block/cnn_block_3/batch_normalization_3/beta/v,Adam/res_block/cnn_block_4/conv2d_4/kernel/v*Adam/res_block/cnn_block_4/conv2d_4/bias/v8Adam/res_block/cnn_block_4/batch_normalization_4/gamma/v7Adam/res_block/cnn_block_4/batch_normalization_4/beta/v,Adam/res_block/cnn_block_5/conv2d_5/kernel/v*Adam/res_block/cnn_block_5/conv2d_5/bias/v8Adam/res_block/cnn_block_5/batch_normalization_5/gamma/v7Adam/res_block/cnn_block_5/batch_normalization_5/beta/v Adam/res_block/conv2d_6/kernel/vAdam/res_block/conv2d_6/bias/v.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/v,Adam/res_block_1/cnn_block_6/conv2d_7/bias/v:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v9Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/v.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/v,Adam/res_block_1/cnn_block_7/conv2d_8/bias/v:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v9Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/v.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/v,Adam/res_block_1/cnn_block_8/conv2d_9/bias/v:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v9Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/v#Adam/res_block_1/conv2d_10/kernel/v!Adam/res_block_1/conv2d_10/bias/v*{
Tint
r2p*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_9290??
?
?
4__inference_batch_normalization_3_layer_call_fn_8271

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_54672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_3_layer_call_fn_8258

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_54362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
&__inference_model_1_layer_call_fn_7685

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_70132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_4_layer_call_fn_8322

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?B
__inference__traced_save_8947
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_res_block_cnn_block_3_conv2d_3_kernel_read_readvariableopB
>savev2_res_block_cnn_block_3_conv2d_3_bias_read_readvariableopP
Lsavev2_res_block_cnn_block_3_batch_normalization_3_gamma_read_readvariableopO
Ksavev2_res_block_cnn_block_3_batch_normalization_3_beta_read_readvariableopD
@savev2_res_block_cnn_block_4_conv2d_4_kernel_read_readvariableopB
>savev2_res_block_cnn_block_4_conv2d_4_bias_read_readvariableopP
Lsavev2_res_block_cnn_block_4_batch_normalization_4_gamma_read_readvariableopO
Ksavev2_res_block_cnn_block_4_batch_normalization_4_beta_read_readvariableopD
@savev2_res_block_cnn_block_5_conv2d_5_kernel_read_readvariableopB
>savev2_res_block_cnn_block_5_conv2d_5_bias_read_readvariableopP
Lsavev2_res_block_cnn_block_5_batch_normalization_5_gamma_read_readvariableopO
Ksavev2_res_block_cnn_block_5_batch_normalization_5_beta_read_readvariableop8
4savev2_res_block_conv2d_6_kernel_read_readvariableop6
2savev2_res_block_conv2d_6_bias_read_readvariableopF
Bsavev2_res_block_1_cnn_block_6_conv2d_7_kernel_read_readvariableopD
@savev2_res_block_1_cnn_block_6_conv2d_7_bias_read_readvariableopR
Nsavev2_res_block_1_cnn_block_6_batch_normalization_6_gamma_read_readvariableopQ
Msavev2_res_block_1_cnn_block_6_batch_normalization_6_beta_read_readvariableopF
Bsavev2_res_block_1_cnn_block_7_conv2d_8_kernel_read_readvariableopD
@savev2_res_block_1_cnn_block_7_conv2d_8_bias_read_readvariableopR
Nsavev2_res_block_1_cnn_block_7_batch_normalization_7_gamma_read_readvariableopQ
Msavev2_res_block_1_cnn_block_7_batch_normalization_7_beta_read_readvariableopF
Bsavev2_res_block_1_cnn_block_8_conv2d_9_kernel_read_readvariableopD
@savev2_res_block_1_cnn_block_8_conv2d_9_bias_read_readvariableopR
Nsavev2_res_block_1_cnn_block_8_batch_normalization_8_gamma_read_readvariableopQ
Msavev2_res_block_1_cnn_block_8_batch_normalization_8_beta_read_readvariableop;
7savev2_res_block_1_conv2d_10_kernel_read_readvariableop9
5savev2_res_block_1_conv2d_10_bias_read_readvariableopV
Rsavev2_res_block_cnn_block_3_batch_normalization_3_moving_mean_read_readvariableopZ
Vsavev2_res_block_cnn_block_3_batch_normalization_3_moving_variance_read_readvariableopV
Rsavev2_res_block_cnn_block_4_batch_normalization_4_moving_mean_read_readvariableopZ
Vsavev2_res_block_cnn_block_4_batch_normalization_4_moving_variance_read_readvariableopV
Rsavev2_res_block_cnn_block_5_batch_normalization_5_moving_mean_read_readvariableopZ
Vsavev2_res_block_cnn_block_5_batch_normalization_5_moving_variance_read_readvariableopX
Tsavev2_res_block_1_cnn_block_6_batch_normalization_6_moving_mean_read_readvariableop\
Xsavev2_res_block_1_cnn_block_6_batch_normalization_6_moving_variance_read_readvariableopX
Tsavev2_res_block_1_cnn_block_7_batch_normalization_7_moving_mean_read_readvariableop\
Xsavev2_res_block_1_cnn_block_7_batch_normalization_7_moving_variance_read_readvariableopX
Tsavev2_res_block_1_cnn_block_8_batch_normalization_8_moving_mean_read_readvariableop\
Xsavev2_res_block_1_cnn_block_8_batch_normalization_8_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_3_conv2d_3_kernel_m_read_readvariableopI
Esavev2_adam_res_block_cnn_block_3_conv2d_3_bias_m_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_3_batch_normalization_3_gamma_m_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_3_batch_normalization_3_beta_m_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_4_conv2d_4_kernel_m_read_readvariableopI
Esavev2_adam_res_block_cnn_block_4_conv2d_4_bias_m_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_4_batch_normalization_4_gamma_m_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_4_batch_normalization_4_beta_m_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_5_conv2d_5_kernel_m_read_readvariableopI
Esavev2_adam_res_block_cnn_block_5_conv2d_5_bias_m_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_5_batch_normalization_5_gamma_m_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_5_batch_normalization_5_beta_m_read_readvariableop?
;savev2_adam_res_block_conv2d_6_kernel_m_read_readvariableop=
9savev2_adam_res_block_conv2d_6_bias_m_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_6_conv2d_7_kernel_m_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_6_conv2d_7_bias_m_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_m_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_m_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_7_conv2d_8_kernel_m_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_7_conv2d_8_bias_m_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_m_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_m_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_8_conv2d_9_kernel_m_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_8_conv2d_9_bias_m_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_m_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_m_read_readvariableopB
>savev2_adam_res_block_1_conv2d_10_kernel_m_read_readvariableop@
<savev2_adam_res_block_1_conv2d_10_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_3_conv2d_3_kernel_v_read_readvariableopI
Esavev2_adam_res_block_cnn_block_3_conv2d_3_bias_v_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_3_batch_normalization_3_gamma_v_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_3_batch_normalization_3_beta_v_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_4_conv2d_4_kernel_v_read_readvariableopI
Esavev2_adam_res_block_cnn_block_4_conv2d_4_bias_v_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_4_batch_normalization_4_gamma_v_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_4_batch_normalization_4_beta_v_read_readvariableopK
Gsavev2_adam_res_block_cnn_block_5_conv2d_5_kernel_v_read_readvariableopI
Esavev2_adam_res_block_cnn_block_5_conv2d_5_bias_v_read_readvariableopW
Ssavev2_adam_res_block_cnn_block_5_batch_normalization_5_gamma_v_read_readvariableopV
Rsavev2_adam_res_block_cnn_block_5_batch_normalization_5_beta_v_read_readvariableop?
;savev2_adam_res_block_conv2d_6_kernel_v_read_readvariableop=
9savev2_adam_res_block_conv2d_6_bias_v_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_6_conv2d_7_kernel_v_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_6_conv2d_7_bias_v_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_v_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_v_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_7_conv2d_8_kernel_v_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_7_conv2d_8_bias_v_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_v_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_v_read_readvariableopM
Isavev2_adam_res_block_1_cnn_block_8_conv2d_9_kernel_v_read_readvariableopK
Gsavev2_adam_res_block_1_cnn_block_8_conv2d_9_bias_v_read_readvariableopY
Usavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_v_read_readvariableopX
Tsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_v_read_readvariableopB
>savev2_adam_res_block_1_conv2d_10_kernel_v_read_readvariableop@
<savev2_adam_res_block_1_conv2d_10_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?9
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?8
value?8B?8pB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?
value?B?pB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?@
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_res_block_cnn_block_3_conv2d_3_kernel_read_readvariableop>savev2_res_block_cnn_block_3_conv2d_3_bias_read_readvariableopLsavev2_res_block_cnn_block_3_batch_normalization_3_gamma_read_readvariableopKsavev2_res_block_cnn_block_3_batch_normalization_3_beta_read_readvariableop@savev2_res_block_cnn_block_4_conv2d_4_kernel_read_readvariableop>savev2_res_block_cnn_block_4_conv2d_4_bias_read_readvariableopLsavev2_res_block_cnn_block_4_batch_normalization_4_gamma_read_readvariableopKsavev2_res_block_cnn_block_4_batch_normalization_4_beta_read_readvariableop@savev2_res_block_cnn_block_5_conv2d_5_kernel_read_readvariableop>savev2_res_block_cnn_block_5_conv2d_5_bias_read_readvariableopLsavev2_res_block_cnn_block_5_batch_normalization_5_gamma_read_readvariableopKsavev2_res_block_cnn_block_5_batch_normalization_5_beta_read_readvariableop4savev2_res_block_conv2d_6_kernel_read_readvariableop2savev2_res_block_conv2d_6_bias_read_readvariableopBsavev2_res_block_1_cnn_block_6_conv2d_7_kernel_read_readvariableop@savev2_res_block_1_cnn_block_6_conv2d_7_bias_read_readvariableopNsavev2_res_block_1_cnn_block_6_batch_normalization_6_gamma_read_readvariableopMsavev2_res_block_1_cnn_block_6_batch_normalization_6_beta_read_readvariableopBsavev2_res_block_1_cnn_block_7_conv2d_8_kernel_read_readvariableop@savev2_res_block_1_cnn_block_7_conv2d_8_bias_read_readvariableopNsavev2_res_block_1_cnn_block_7_batch_normalization_7_gamma_read_readvariableopMsavev2_res_block_1_cnn_block_7_batch_normalization_7_beta_read_readvariableopBsavev2_res_block_1_cnn_block_8_conv2d_9_kernel_read_readvariableop@savev2_res_block_1_cnn_block_8_conv2d_9_bias_read_readvariableopNsavev2_res_block_1_cnn_block_8_batch_normalization_8_gamma_read_readvariableopMsavev2_res_block_1_cnn_block_8_batch_normalization_8_beta_read_readvariableop7savev2_res_block_1_conv2d_10_kernel_read_readvariableop5savev2_res_block_1_conv2d_10_bias_read_readvariableopRsavev2_res_block_cnn_block_3_batch_normalization_3_moving_mean_read_readvariableopVsavev2_res_block_cnn_block_3_batch_normalization_3_moving_variance_read_readvariableopRsavev2_res_block_cnn_block_4_batch_normalization_4_moving_mean_read_readvariableopVsavev2_res_block_cnn_block_4_batch_normalization_4_moving_variance_read_readvariableopRsavev2_res_block_cnn_block_5_batch_normalization_5_moving_mean_read_readvariableopVsavev2_res_block_cnn_block_5_batch_normalization_5_moving_variance_read_readvariableopTsavev2_res_block_1_cnn_block_6_batch_normalization_6_moving_mean_read_readvariableopXsavev2_res_block_1_cnn_block_6_batch_normalization_6_moving_variance_read_readvariableopTsavev2_res_block_1_cnn_block_7_batch_normalization_7_moving_mean_read_readvariableopXsavev2_res_block_1_cnn_block_7_batch_normalization_7_moving_variance_read_readvariableopTsavev2_res_block_1_cnn_block_8_batch_normalization_8_moving_mean_read_readvariableopXsavev2_res_block_1_cnn_block_8_batch_normalization_8_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopGsavev2_adam_res_block_cnn_block_3_conv2d_3_kernel_m_read_readvariableopEsavev2_adam_res_block_cnn_block_3_conv2d_3_bias_m_read_readvariableopSsavev2_adam_res_block_cnn_block_3_batch_normalization_3_gamma_m_read_readvariableopRsavev2_adam_res_block_cnn_block_3_batch_normalization_3_beta_m_read_readvariableopGsavev2_adam_res_block_cnn_block_4_conv2d_4_kernel_m_read_readvariableopEsavev2_adam_res_block_cnn_block_4_conv2d_4_bias_m_read_readvariableopSsavev2_adam_res_block_cnn_block_4_batch_normalization_4_gamma_m_read_readvariableopRsavev2_adam_res_block_cnn_block_4_batch_normalization_4_beta_m_read_readvariableopGsavev2_adam_res_block_cnn_block_5_conv2d_5_kernel_m_read_readvariableopEsavev2_adam_res_block_cnn_block_5_conv2d_5_bias_m_read_readvariableopSsavev2_adam_res_block_cnn_block_5_batch_normalization_5_gamma_m_read_readvariableopRsavev2_adam_res_block_cnn_block_5_batch_normalization_5_beta_m_read_readvariableop;savev2_adam_res_block_conv2d_6_kernel_m_read_readvariableop9savev2_adam_res_block_conv2d_6_bias_m_read_readvariableopIsavev2_adam_res_block_1_cnn_block_6_conv2d_7_kernel_m_read_readvariableopGsavev2_adam_res_block_1_cnn_block_6_conv2d_7_bias_m_read_readvariableopUsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_m_read_readvariableopTsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_m_read_readvariableopIsavev2_adam_res_block_1_cnn_block_7_conv2d_8_kernel_m_read_readvariableopGsavev2_adam_res_block_1_cnn_block_7_conv2d_8_bias_m_read_readvariableopUsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_m_read_readvariableopTsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_m_read_readvariableopIsavev2_adam_res_block_1_cnn_block_8_conv2d_9_kernel_m_read_readvariableopGsavev2_adam_res_block_1_cnn_block_8_conv2d_9_bias_m_read_readvariableopUsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_m_read_readvariableopTsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_m_read_readvariableop>savev2_adam_res_block_1_conv2d_10_kernel_m_read_readvariableop<savev2_adam_res_block_1_conv2d_10_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopGsavev2_adam_res_block_cnn_block_3_conv2d_3_kernel_v_read_readvariableopEsavev2_adam_res_block_cnn_block_3_conv2d_3_bias_v_read_readvariableopSsavev2_adam_res_block_cnn_block_3_batch_normalization_3_gamma_v_read_readvariableopRsavev2_adam_res_block_cnn_block_3_batch_normalization_3_beta_v_read_readvariableopGsavev2_adam_res_block_cnn_block_4_conv2d_4_kernel_v_read_readvariableopEsavev2_adam_res_block_cnn_block_4_conv2d_4_bias_v_read_readvariableopSsavev2_adam_res_block_cnn_block_4_batch_normalization_4_gamma_v_read_readvariableopRsavev2_adam_res_block_cnn_block_4_batch_normalization_4_beta_v_read_readvariableopGsavev2_adam_res_block_cnn_block_5_conv2d_5_kernel_v_read_readvariableopEsavev2_adam_res_block_cnn_block_5_conv2d_5_bias_v_read_readvariableopSsavev2_adam_res_block_cnn_block_5_batch_normalization_5_gamma_v_read_readvariableopRsavev2_adam_res_block_cnn_block_5_batch_normalization_5_beta_v_read_readvariableop;savev2_adam_res_block_conv2d_6_kernel_v_read_readvariableop9savev2_adam_res_block_conv2d_6_bias_v_read_readvariableopIsavev2_adam_res_block_1_cnn_block_6_conv2d_7_kernel_v_read_readvariableopGsavev2_adam_res_block_1_cnn_block_6_conv2d_7_bias_v_read_readvariableopUsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_v_read_readvariableopTsavev2_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_v_read_readvariableopIsavev2_adam_res_block_1_cnn_block_7_conv2d_8_kernel_v_read_readvariableopGsavev2_adam_res_block_1_cnn_block_7_conv2d_8_bias_v_read_readvariableopUsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_v_read_readvariableopTsavev2_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_v_read_readvariableopIsavev2_adam_res_block_1_cnn_block_8_conv2d_9_kernel_v_read_readvariableopGsavev2_adam_res_block_1_cnn_block_8_conv2d_9_bias_v_read_readvariableopUsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_v_read_readvariableopTsavev2_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_v_read_readvariableop>savev2_adam_res_block_1_conv2d_10_kernel_v_read_readvariableop<savev2_adam_res_block_1_conv2d_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *~
dtypest
r2p	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??
:
: : : : : : : : : :  : : : : @:@:@:@: : :@?:?:?:?:??:?:?:?:??:?:?:?:@?:?: : : : :@:@:?:?:?:?:?:?: : : : :
??
:
: : : : :  : : : : @:@:@:@: : :@?:?:?:?:??:?:?:?:??:?:?:?:@?:?:
??
:
: : : : :  : : : : @:@:@:@: : :@?:?:?:?:??:?:?:?:??:?:?:?:@?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:! 

_output_shapes	
:?:!!

_output_shapes	
:?:-")
'
_output_shapes
:@?:!#

_output_shapes	
:?: $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
:@: )

_output_shapes
:@:!*

_output_shapes	
:?:!+

_output_shapes	
:?:!,

_output_shapes	
:?:!-

_output_shapes	
:?:!.

_output_shapes	
:?:!/

_output_shapes	
:?:0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :&4"
 
_output_shapes
:
??
: 5

_output_shapes
:
:,6(
&
_output_shapes
: : 7

_output_shapes
: : 8

_output_shapes
: : 9

_output_shapes
: :,:(
&
_output_shapes
:  : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: :,>(
&
_output_shapes
: @: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:,B(
&
_output_shapes
: : C

_output_shapes
: :-D)
'
_output_shapes
:@?:!E

_output_shapes	
:?:!F

_output_shapes	
:?:!G

_output_shapes	
:?:.H*
(
_output_shapes
:??:!I

_output_shapes	
:?:!J

_output_shapes	
:?:!K

_output_shapes	
:?:.L*
(
_output_shapes
:??:!M

_output_shapes	
:?:!N

_output_shapes	
:?:!O

_output_shapes	
:?:-P)
'
_output_shapes
:@?:!Q

_output_shapes	
:?:&R"
 
_output_shapes
:
??
: S

_output_shapes
:
:,T(
&
_output_shapes
: : U

_output_shapes
: : V

_output_shapes
: : W

_output_shapes
: :,X(
&
_output_shapes
:  : Y

_output_shapes
: : Z

_output_shapes
: : [

_output_shapes
: :,\(
&
_output_shapes
: @: ]

_output_shapes
:@: ^

_output_shapes
:@: _

_output_shapes
:@:,`(
&
_output_shapes
: : a

_output_shapes
: :-b)
'
_output_shapes
:@?:!c

_output_shapes	
:?:!d

_output_shapes	
:?:!e

_output_shapes	
:?:.f*
(
_output_shapes
:??:!g

_output_shapes	
:?:!h

_output_shapes	
:?:!i

_output_shapes	
:?:.j*
(
_output_shapes
:??:!k

_output_shapes	
:?:!l

_output_shapes	
:?:!m

_output_shapes	
:?:-n)
'
_output_shapes
:@?:!o

_output_shapes	
:?:p

_output_shapes
: 
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5675

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8355

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8245

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8483

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
ۗ
?
E__inference_res_block_1_layer_call_and_return_conditional_losses_6396
input_tensor7
3cnn_block_6_conv2d_7_conv2d_readvariableop_resource8
4cnn_block_6_conv2d_7_biasadd_readvariableop_resource=
9cnn_block_6_batch_normalization_6_readvariableop_resource?
;cnn_block_6_batch_normalization_6_readvariableop_1_resourceN
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_7_conv2d_8_conv2d_readvariableop_resource8
4cnn_block_7_conv2d_8_biasadd_readvariableop_resource=
9cnn_block_7_batch_normalization_7_readvariableop_resource?
;cnn_block_7_batch_normalization_7_readvariableop_1_resourceN
Jcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource7
3cnn_block_8_conv2d_9_conv2d_readvariableop_resource8
4cnn_block_8_conv2d_9_biasadd_readvariableop_resource=
9cnn_block_8_batch_normalization_8_readvariableop_resource?
;cnn_block_8_batch_normalization_8_readvariableop_1_resourceN
Jcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??0cnn_block_6/batch_normalization_6/AssignNewValue?2cnn_block_6/batch_normalization_6/AssignNewValue_1?Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_6/batch_normalization_6/ReadVariableOp?2cnn_block_6/batch_normalization_6/ReadVariableOp_1?+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?0cnn_block_7/batch_normalization_7/AssignNewValue?2cnn_block_7/batch_normalization_7/AssignNewValue_1?Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_7/batch_normalization_7/ReadVariableOp?2cnn_block_7/batch_normalization_7/ReadVariableOp_1?+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?0cnn_block_8/batch_normalization_8/AssignNewValue?2cnn_block_8/batch_normalization_8/AssignNewValue_1?Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_8/batch_normalization_8/ReadVariableOp?2cnn_block_8/batch_normalization_8/ReadVariableOp_1?+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?
cnn_block_6/conv2d_7/Conv2DConv2Dinput_tensor2cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_6/conv2d_7/Conv2D?
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?
cnn_block_6/conv2d_7/BiasAddBiasAdd$cnn_block_6/conv2d_7/Conv2D:output:03cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_6/conv2d_7/BiasAdd?
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_6/batch_normalization_6/ReadVariableOp?
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_6/batch_normalization_6/ReadVariableOp_1?
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_7/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_6/batch_normalization_6/FusedBatchNormV3?
0cnn_block_6/batch_normalization_6/AssignNewValueAssignVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource?cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_6/batch_normalization_6/AssignNewValue?
2cnn_block_6/batch_normalization_6/AssignNewValue_1AssignVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0D^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_6/batch_normalization_6/AssignNewValue_1?
cnn_block_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_6/Relu?
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?
cnn_block_7/conv2d_8/Conv2DConv2Dcnn_block_6/Relu:activations:02cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_7/conv2d_8/Conv2D?
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?
cnn_block_7/conv2d_8/BiasAddBiasAdd$cnn_block_7/conv2d_8/Conv2D:output:03cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_7/conv2d_8/BiasAdd?
0cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_7/batch_normalization_7/ReadVariableOp?
2cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_7/batch_normalization_7/ReadVariableOp_1?
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_7/conv2d_8/BiasAdd:output:08cnn_block_7/batch_normalization_7/ReadVariableOp:value:0:cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_7/batch_normalization_7/FusedBatchNormV3?
0cnn_block_7/batch_normalization_7/AssignNewValueAssignVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource?cnn_block_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0B^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_7/batch_normalization_7/AssignNewValue?
2cnn_block_7/batch_normalization_7/AssignNewValue_1AssignVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0D^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_7/batch_normalization_7/AssignNewValue_1?
cnn_block_7/ReluRelu6cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_7/Relu?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinput_tensor'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
conv2d_10/BiasAdd?
addAddV2cnn_block_7/Relu:activations:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????OO?2
add?
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?
cnn_block_8/conv2d_9/Conv2DConv2Dadd:z:02cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_8/conv2d_9/Conv2D?
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?
cnn_block_8/conv2d_9/BiasAddBiasAdd$cnn_block_8/conv2d_9/Conv2D:output:03cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_8/conv2d_9/BiasAdd?
0cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_8/batch_normalization_8/ReadVariableOp?
2cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_8/batch_normalization_8/ReadVariableOp_1?
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_8/conv2d_9/BiasAdd:output:08cnn_block_8/batch_normalization_8/ReadVariableOp:value:0:cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_8/batch_normalization_8/FusedBatchNormV3?
0cnn_block_8/batch_normalization_8/AssignNewValueAssignVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource?cnn_block_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0B^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_8/batch_normalization_8/AssignNewValue?
2cnn_block_8/batch_normalization_8/AssignNewValue_1AssignVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0D^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_8/batch_normalization_8/AssignNewValue_1?
cnn_block_8/ReluRelu6cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_8/Relu?
max_pooling2d_1/MaxPoolMaxPoolcnn_block_8/Relu:activations:0*0
_output_shapes
:?????????''?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
IdentityIdentity max_pooling2d_1/MaxPool:output:01^cnn_block_6/batch_normalization_6/AssignNewValue3^cnn_block_6/batch_normalization_6/AssignNewValue_1B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^cnn_block_6/batch_normalization_6/ReadVariableOp3^cnn_block_6/batch_normalization_6/ReadVariableOp_1,^cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+^cnn_block_6/conv2d_7/Conv2D/ReadVariableOp1^cnn_block_7/batch_normalization_7/AssignNewValue3^cnn_block_7/batch_normalization_7/AssignNewValue_1B^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^cnn_block_7/batch_normalization_7/ReadVariableOp3^cnn_block_7/batch_normalization_7/ReadVariableOp_1,^cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+^cnn_block_7/conv2d_8/Conv2D/ReadVariableOp1^cnn_block_8/batch_normalization_8/AssignNewValue3^cnn_block_8/batch_normalization_8/AssignNewValue_1B^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^cnn_block_8/batch_normalization_8/ReadVariableOp3^cnn_block_8/batch_normalization_8/ReadVariableOp_1,^cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+^cnn_block_8/conv2d_9/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????''?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????OO@::::::::::::::::::::2d
0cnn_block_6/batch_normalization_6/AssignNewValue0cnn_block_6/batch_normalization_6/AssignNewValue2h
2cnn_block_6/batch_normalization_6/AssignNewValue_12cnn_block_6/batch_normalization_6/AssignNewValue_12?
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_6/batch_normalization_6/ReadVariableOp0cnn_block_6/batch_normalization_6/ReadVariableOp2h
2cnn_block_6/batch_normalization_6/ReadVariableOp_12cnn_block_6/batch_normalization_6/ReadVariableOp_12Z
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2X
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2d
0cnn_block_7/batch_normalization_7/AssignNewValue0cnn_block_7/batch_normalization_7/AssignNewValue2h
2cnn_block_7/batch_normalization_7/AssignNewValue_12cnn_block_7/batch_normalization_7/AssignNewValue_12?
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_7/batch_normalization_7/ReadVariableOp0cnn_block_7/batch_normalization_7/ReadVariableOp2h
2cnn_block_7/batch_normalization_7/ReadVariableOp_12cnn_block_7/batch_normalization_7/ReadVariableOp_12Z
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2X
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2d
0cnn_block_8/batch_normalization_8/AssignNewValue0cnn_block_8/batch_normalization_8/AssignNewValue2h
2cnn_block_8/batch_normalization_8/AssignNewValue_12cnn_block_8/batch_normalization_8/AssignNewValue_12?
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpAcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_8/batch_normalization_8/ReadVariableOp0cnn_block_8/batch_normalization_8/ReadVariableOp2h
2cnn_block_8/batch_normalization_8/ReadVariableOp_12cnn_block_8/batch_normalization_8/ReadVariableOp_12Z
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2X
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:?????????OO@
&
_user_specified_nameinput_tensor
??
?2
__inference__wrapped_model_5374
input_1I
Emodel_1_res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resourceJ
Fmodel_1_res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resourceO
Kmodel_1_res_block_cnn_block_3_batch_normalization_3_readvariableop_resourceQ
Mmodel_1_res_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource`
\model_1_res_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceb
^model_1_res_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceI
Emodel_1_res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resourceJ
Fmodel_1_res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resourceO
Kmodel_1_res_block_cnn_block_4_batch_normalization_4_readvariableop_resourceQ
Mmodel_1_res_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource`
\model_1_res_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceb
^model_1_res_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource=
9model_1_res_block_conv2d_6_conv2d_readvariableop_resource>
:model_1_res_block_conv2d_6_biasadd_readvariableop_resourceI
Emodel_1_res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resourceJ
Fmodel_1_res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resourceO
Kmodel_1_res_block_cnn_block_5_batch_normalization_5_readvariableop_resourceQ
Mmodel_1_res_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource`
\model_1_res_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceb
^model_1_res_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceK
Gmodel_1_res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resourceL
Hmodel_1_res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resourceQ
Mmodel_1_res_block_1_cnn_block_6_batch_normalization_6_readvariableop_resourceS
Omodel_1_res_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resourceb
^model_1_res_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourced
`model_1_res_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceK
Gmodel_1_res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resourceL
Hmodel_1_res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resourceQ
Mmodel_1_res_block_1_cnn_block_7_batch_normalization_7_readvariableop_resourceS
Omodel_1_res_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resourceb
^model_1_res_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resourced
`model_1_res_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource@
<model_1_res_block_1_conv2d_10_conv2d_readvariableop_resourceA
=model_1_res_block_1_conv2d_10_biasadd_readvariableop_resourceK
Gmodel_1_res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resourceL
Hmodel_1_res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resourceQ
Mmodel_1_res_block_1_cnn_block_8_batch_normalization_8_readvariableop_resourceS
Omodel_1_res_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resourceb
^model_1_res_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourced
`model_1_res_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource2
.model_1_dense_2_matmul_readvariableop_resource3
/model_1_dense_2_biasadd_readvariableop_resource
identity??&model_1/dense_2/BiasAdd/ReadVariableOp?%model_1/dense_2/MatMul/ReadVariableOp?Smodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?Bmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp?Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1?=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?Smodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?Bmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp?Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1?=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?Smodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?Bmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp?Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1?=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp?0model_1/res_block/conv2d_6/Conv2D/ReadVariableOp?Umodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?Dmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1??model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?Umodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?Dmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1??model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?Umodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?Dmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1??model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp?3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp?
<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpEmodel_1_res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?
-model_1/res_block/cnn_block_3/conv2d_3/Conv2DConv2Dinput_1Dmodel_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2/
-model_1/res_block/cnn_block_3/conv2d_3/Conv2D?
=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?
.model_1/res_block/cnn_block_3/conv2d_3/BiasAddBiasAdd6model_1/res_block/cnn_block_3/conv2d_3/Conv2D:output:0Emodel_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 20
.model_1/res_block/cnn_block_3/conv2d_3/BiasAdd?
Bmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpKmodel_1_res_block_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp?
Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpMmodel_1_res_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1?
Smodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp\model_1_res_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02U
Smodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^model_1_res_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02W
Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
Dmodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV37model_1/res_block/cnn_block_3/conv2d_3/BiasAdd:output:0Jmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Lmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0[model_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0]model_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2F
Dmodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3?
"model_1/res_block/cnn_block_3/ReluReluHmodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2$
"model_1/res_block/cnn_block_3/Relu?
<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpEmodel_1_res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02>
<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?
-model_1/res_block/cnn_block_4/conv2d_4/Conv2DConv2D0model_1/res_block/cnn_block_3/Relu:activations:0Dmodel_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2/
-model_1/res_block/cnn_block_4/conv2d_4/Conv2D?
=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?
.model_1/res_block/cnn_block_4/conv2d_4/BiasAddBiasAdd6model_1/res_block/cnn_block_4/conv2d_4/Conv2D:output:0Emodel_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 20
.model_1/res_block/cnn_block_4/conv2d_4/BiasAdd?
Bmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpKmodel_1_res_block_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp?
Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpMmodel_1_res_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1?
Smodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp\model_1_res_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02U
Smodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^model_1_res_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02W
Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
Dmodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV37model_1/res_block/cnn_block_4/conv2d_4/BiasAdd:output:0Jmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Lmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0[model_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0]model_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2F
Dmodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3?
"model_1/res_block/cnn_block_4/ReluReluHmodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2$
"model_1/res_block/cnn_block_4/Relu?
0model_1/res_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp9model_1_res_block_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype022
0model_1/res_block/conv2d_6/Conv2D/ReadVariableOp?
!model_1/res_block/conv2d_6/Conv2DConv2Dinput_18model_1/res_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2#
!model_1/res_block/conv2d_6/Conv2D?
1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp:model_1_res_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp?
"model_1/res_block/conv2d_6/BiasAddBiasAdd*model_1/res_block/conv2d_6/Conv2D:output:09model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2$
"model_1/res_block/conv2d_6/BiasAdd?
model_1/res_block/addAddV20model_1/res_block/cnn_block_4/Relu:activations:0+model_1/res_block/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
model_1/res_block/add?
<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpEmodel_1_res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02>
<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?
-model_1/res_block/cnn_block_5/conv2d_5/Conv2DConv2Dmodel_1/res_block/add:z:0Dmodel_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2/
-model_1/res_block/cnn_block_5/conv2d_5/Conv2D?
=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02?
=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?
.model_1/res_block/cnn_block_5/conv2d_5/BiasAddBiasAdd6model_1/res_block/cnn_block_5/conv2d_5/Conv2D:output:0Emodel_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@20
.model_1/res_block/cnn_block_5/conv2d_5/BiasAdd?
Bmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpKmodel_1_res_block_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp?
Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpMmodel_1_res_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1?
Smodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp\model_1_res_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02U
Smodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^model_1_res_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02W
Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
Dmodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV37model_1/res_block/cnn_block_5/conv2d_5/BiasAdd:output:0Jmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Lmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0[model_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0]model_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2F
Dmodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3?
"model_1/res_block/cnn_block_5/ReluReluHmodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2$
"model_1/res_block/cnn_block_5/Relu?
'model_1/res_block/max_pooling2d/MaxPoolMaxPool0model_1/res_block/cnn_block_5/Relu:activations:0*/
_output_shapes
:?????????OO@*
ksize
*
paddingVALID*
strides
2)
'model_1/res_block/max_pooling2d/MaxPool?
>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOpGmodel_1_res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02@
>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?
/model_1/res_block_1/cnn_block_6/conv2d_7/Conv2DConv2D0model_1/res_block/max_pooling2d/MaxPool:output:0Fmodel_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
21
/model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D?
?model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpHmodel_1_res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?
0model_1/res_block_1/cnn_block_6/conv2d_7/BiasAddBiasAdd8model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D:output:0Gmodel_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?22
0model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd?
Dmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpMmodel_1_res_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?
Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpOmodel_1_res_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1?
Umodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp^model_1_res_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02W
Umodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`model_1_res_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Y
Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV39model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd:output:0Lmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Nmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0]model_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0_model_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2H
Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3?
$model_1/res_block_1/cnn_block_6/ReluReluJmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2&
$model_1/res_block_1/cnn_block_6/Relu?
>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOpGmodel_1_res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02@
>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?
/model_1/res_block_1/cnn_block_7/conv2d_8/Conv2DConv2D2model_1/res_block_1/cnn_block_6/Relu:activations:0Fmodel_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
21
/model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D?
?model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpHmodel_1_res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?
0model_1/res_block_1/cnn_block_7/conv2d_8/BiasAddBiasAdd8model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D:output:0Gmodel_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?22
0model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd?
Dmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOpMmodel_1_res_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?
Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOpOmodel_1_res_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1?
Umodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp^model_1_res_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02W
Umodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`model_1_res_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Y
Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV39model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd:output:0Lmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp:value:0Nmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0]model_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0_model_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2H
Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3?
$model_1/res_block_1/cnn_block_7/ReluReluJmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2&
$model_1/res_block_1/cnn_block_7/Relu?
3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp<model_1_res_block_1_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp?
$model_1/res_block_1/conv2d_10/Conv2DConv2D0model_1/res_block/max_pooling2d/MaxPool:output:0;model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2&
$model_1/res_block_1/conv2d_10/Conv2D?
4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp=model_1_res_block_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype026
4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp?
%model_1/res_block_1/conv2d_10/BiasAddBiasAdd-model_1/res_block_1/conv2d_10/Conv2D:output:0<model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2'
%model_1/res_block_1/conv2d_10/BiasAdd?
model_1/res_block_1/addAddV22model_1/res_block_1/cnn_block_7/Relu:activations:0.model_1/res_block_1/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????OO?2
model_1/res_block_1/add?
>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOpGmodel_1_res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02@
>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?
/model_1/res_block_1/cnn_block_8/conv2d_9/Conv2DConv2Dmodel_1/res_block_1/add:z:0Fmodel_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
21
/model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D?
?model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpHmodel_1_res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?
0model_1/res_block_1/cnn_block_8/conv2d_9/BiasAddBiasAdd8model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D:output:0Gmodel_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?22
0model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd?
Dmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOpMmodel_1_res_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?
Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOpOmodel_1_res_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1?
Umodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp^model_1_res_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02W
Umodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`model_1_res_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Y
Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV39model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd:output:0Lmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp:value:0Nmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0]model_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0_model_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2H
Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3?
$model_1/res_block_1/cnn_block_8/ReluReluJmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2&
$model_1/res_block_1/cnn_block_8/Relu?
+model_1/res_block_1/max_pooling2d_1/MaxPoolMaxPool2model_1/res_block_1/cnn_block_8/Relu:activations:0*0
_output_shapes
:?????????''?*
ksize
*
paddingVALID*
strides
2-
+model_1/res_block_1/max_pooling2d_1/MaxPool?
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
model_1/flatten_1/Const?
model_1/flatten_1/ReshapeReshape4model_1/res_block_1/max_pooling2d_1/MaxPool:output:0 model_1/flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
model_1/flatten_1/Reshape?
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02'
%model_1/dense_2/MatMul/ReadVariableOp?
model_1/dense_2/MatMulMatMul"model_1/flatten_1/Reshape:output:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_1/dense_2/MatMul?
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&model_1/dense_2/BiasAdd/ReadVariableOp?
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_1/dense_2/BiasAdd?
IdentityIdentity model_1/dense_2/BiasAdd:output:0'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOpT^model_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpV^model_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1C^model_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOpE^model_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1>^model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp=^model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpT^model_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpV^model_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1C^model_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOpE^model_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1>^model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp=^model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpT^model_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpV^model_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1C^model_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOpE^model_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1>^model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp=^model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2^model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp1^model_1/res_block/conv2d_6/Conv2D/ReadVariableOpV^model_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpX^model_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1E^model_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpG^model_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1@^model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?^model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpV^model_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpX^model_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1E^model_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpG^model_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1@^model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?^model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpV^model_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpX^model_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1E^model_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpG^model_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1@^model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?^model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp5^model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp4^model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp2?
Smodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpSmodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Umodel_1/res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12?
Bmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOpBmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp2?
Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1Dmodel_1/res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_12~
=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp=model_1/res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2|
<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp<model_1/res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2?
Smodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpSmodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Umodel_1/res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12?
Bmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOpBmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp2?
Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1Dmodel_1/res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_12~
=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp=model_1/res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2|
<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp<model_1/res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2?
Smodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpSmodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Umodel_1/res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12?
Bmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOpBmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp2?
Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1Dmodel_1/res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_12~
=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp=model_1/res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2|
<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp<model_1/res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2f
1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp1model_1/res_block/conv2d_6/BiasAdd/ReadVariableOp2d
0model_1/res_block/conv2d_6/Conv2D/ReadVariableOp0model_1/res_block/conv2d_6/Conv2D/ReadVariableOp2?
Umodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpUmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Wmodel_1/res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12?
Dmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpDmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp2?
Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1Fmodel_1/res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_12?
?model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?model_1/res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2?
>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp>model_1/res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2?
Umodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpUmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Wmodel_1/res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12?
Dmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpDmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp2?
Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1Fmodel_1/res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_12?
?model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?model_1/res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2?
>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp>model_1/res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2?
Umodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpUmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Wmodel_1/res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12?
Dmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpDmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp2?
Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1Fmodel_1/res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_12?
?model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?model_1/res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2?
>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp>model_1/res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2l
4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp4model_1/res_block_1/conv2d_10/BiasAdd/ReadVariableOp2j
3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp3model_1/res_block_1/conv2d_10/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5467

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_6_layer_call_fn_8450

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5760

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8373

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?u
?
E__inference_res_block_1_layer_call_and_return_conditional_losses_8087
input_tensor7
3cnn_block_6_conv2d_7_conv2d_readvariableop_resource8
4cnn_block_6_conv2d_7_biasadd_readvariableop_resource=
9cnn_block_6_batch_normalization_6_readvariableop_resource?
;cnn_block_6_batch_normalization_6_readvariableop_1_resourceN
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_7_conv2d_8_conv2d_readvariableop_resource8
4cnn_block_7_conv2d_8_biasadd_readvariableop_resource=
9cnn_block_7_batch_normalization_7_readvariableop_resource?
;cnn_block_7_batch_normalization_7_readvariableop_1_resourceN
Jcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource7
3cnn_block_8_conv2d_9_conv2d_readvariableop_resource8
4cnn_block_8_conv2d_9_biasadd_readvariableop_resource=
9cnn_block_8_batch_normalization_8_readvariableop_resource?
;cnn_block_8_batch_normalization_8_readvariableop_1_resourceN
Jcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_6/batch_normalization_6/ReadVariableOp?2cnn_block_6/batch_normalization_6/ReadVariableOp_1?+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_7/batch_normalization_7/ReadVariableOp?2cnn_block_7/batch_normalization_7/ReadVariableOp_1?+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_8/batch_normalization_8/ReadVariableOp?2cnn_block_8/batch_normalization_8/ReadVariableOp_1?+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?
cnn_block_6/conv2d_7/Conv2DConv2Dinput_tensor2cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_6/conv2d_7/Conv2D?
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?
cnn_block_6/conv2d_7/BiasAddBiasAdd$cnn_block_6/conv2d_7/Conv2D:output:03cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_6/conv2d_7/BiasAdd?
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_6/batch_normalization_6/ReadVariableOp?
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_6/batch_normalization_6/ReadVariableOp_1?
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_7/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 24
2cnn_block_6/batch_normalization_6/FusedBatchNormV3?
cnn_block_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_6/Relu?
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?
cnn_block_7/conv2d_8/Conv2DConv2Dcnn_block_6/Relu:activations:02cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_7/conv2d_8/Conv2D?
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?
cnn_block_7/conv2d_8/BiasAddBiasAdd$cnn_block_7/conv2d_8/Conv2D:output:03cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_7/conv2d_8/BiasAdd?
0cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_7/batch_normalization_7/ReadVariableOp?
2cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_7/batch_normalization_7/ReadVariableOp_1?
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_7/conv2d_8/BiasAdd:output:08cnn_block_7/batch_normalization_7/ReadVariableOp:value:0:cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 24
2cnn_block_7/batch_normalization_7/FusedBatchNormV3?
cnn_block_7/ReluRelu6cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_7/Relu?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinput_tensor'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
conv2d_10/BiasAdd?
addAddV2cnn_block_7/Relu:activations:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????OO?2
add?
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?
cnn_block_8/conv2d_9/Conv2DConv2Dadd:z:02cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_8/conv2d_9/Conv2D?
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?
cnn_block_8/conv2d_9/BiasAddBiasAdd$cnn_block_8/conv2d_9/Conv2D:output:03cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_8/conv2d_9/BiasAdd?
0cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_8/batch_normalization_8/ReadVariableOp?
2cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_8/batch_normalization_8/ReadVariableOp_1?
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_8/conv2d_9/BiasAdd:output:08cnn_block_8/batch_normalization_8/ReadVariableOp:value:0:cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 24
2cnn_block_8/batch_normalization_8/FusedBatchNormV3?
cnn_block_8/ReluRelu6cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_8/Relu?
max_pooling2d_1/MaxPoolMaxPoolcnn_block_8/Relu:activations:0*0
_output_shapes
:?????????''?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?	
IdentityIdentity max_pooling2d_1/MaxPool:output:0B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^cnn_block_6/batch_normalization_6/ReadVariableOp3^cnn_block_6/batch_normalization_6/ReadVariableOp_1,^cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+^cnn_block_6/conv2d_7/Conv2D/ReadVariableOpB^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^cnn_block_7/batch_normalization_7/ReadVariableOp3^cnn_block_7/batch_normalization_7/ReadVariableOp_1,^cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+^cnn_block_7/conv2d_8/Conv2D/ReadVariableOpB^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^cnn_block_8/batch_normalization_8/ReadVariableOp3^cnn_block_8/batch_normalization_8/ReadVariableOp_1,^cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+^cnn_block_8/conv2d_9/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????''?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????OO@::::::::::::::::::::2?
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_6/batch_normalization_6/ReadVariableOp0cnn_block_6/batch_normalization_6/ReadVariableOp2h
2cnn_block_6/batch_normalization_6/ReadVariableOp_12cnn_block_6/batch_normalization_6/ReadVariableOp_12Z
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2X
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2?
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_7/batch_normalization_7/ReadVariableOp0cnn_block_7/batch_normalization_7/ReadVariableOp2h
2cnn_block_7/batch_normalization_7/ReadVariableOp_12cnn_block_7/batch_normalization_7/ReadVariableOp_12Z
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2X
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2?
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpAcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_8/batch_normalization_8/ReadVariableOp0cnn_block_8/batch_normalization_8/ReadVariableOp2h
2cnn_block_8/batch_normalization_8/ReadVariableOp_12cnn_block_8/batch_normalization_8/ReadVariableOp_12Z
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2X
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:?????????OO@
&
_user_specified_nameinput_tensor
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5999

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?u
?
C__inference_res_block_layer_call_and_return_conditional_losses_6182
input_tensor7
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource8
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource=
9cnn_block_3_batch_normalization_3_readvariableop_resource?
;cnn_block_3_batch_normalization_3_readvariableop_1_resourceN
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource8
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource=
9cnn_block_4_batch_normalization_4_readvariableop_resource?
;cnn_block_4_batch_normalization_4_readvariableop_1_resourceN
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource7
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource8
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource=
9cnn_block_5_batch_normalization_5_readvariableop_resource?
;cnn_block_5_batch_normalization_5_readvariableop_1_resourceN
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity??Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_3/batch_normalization_3/ReadVariableOp?2cnn_block_3/batch_normalization_3/ReadVariableOp_1?+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_4/batch_normalization_4/ReadVariableOp?2cnn_block_4/batch_normalization_4/ReadVariableOp_1?+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_5/batch_normalization_5/ReadVariableOp?2cnn_block_5/batch_normalization_5/ReadVariableOp_1?+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?
cnn_block_3/conv2d_3/Conv2DConv2Dinput_tensor2cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
cnn_block_3/conv2d_3/Conv2D?
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_3/conv2d_3/BiasAdd?
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_3/batch_normalization_3/ReadVariableOp?
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_3/batch_normalization_3/ReadVariableOp_1?
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 24
2cnn_block_3/batch_normalization_3/FusedBatchNormV3?
cnn_block_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_3/Relu?
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?
cnn_block_4/conv2d_4/Conv2DConv2Dcnn_block_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
cnn_block_4/conv2d_4/Conv2D?
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_4/conv2d_4/BiasAdd?
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_4/batch_normalization_4/ReadVariableOp?
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_4/batch_normalization_4/ReadVariableOp_1?
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 24
2cnn_block_4/batch_normalization_4/FusedBatchNormV3?
cnn_block_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_4/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_6/BiasAdd?
addAddV2cnn_block_4/Relu:activations:0conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
add?
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?
cnn_block_5/conv2d_5/Conv2DConv2Dadd:z:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
cnn_block_5/conv2d_5/Conv2D?
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
cnn_block_5/conv2d_5/BiasAdd?
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_5/batch_normalization_5/ReadVariableOp?
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_5/batch_normalization_5/ReadVariableOp_1?
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2cnn_block_5/batch_normalization_5/FusedBatchNormV3?
cnn_block_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
cnn_block_5/Relu?
max_pooling2d/MaxPoolMaxPoolcnn_block_5/Relu:activations:0*/
_output_shapes
:?????????OO@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?	
IdentityIdentitymax_pooling2d/MaxPool:output:0B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^cnn_block_3/batch_normalization_3/ReadVariableOp3^cnn_block_3/batch_normalization_3/ReadVariableOp_1,^cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+^cnn_block_3/conv2d_3/Conv2D/ReadVariableOpB^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^cnn_block_4/batch_normalization_4/ReadVariableOp3^cnn_block_4/batch_normalization_4/ReadVariableOp_1,^cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+^cnn_block_4/conv2d_4/Conv2D/ReadVariableOpB^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^cnn_block_5/batch_normalization_5/ReadVariableOp3^cnn_block_5/batch_normalization_5/ReadVariableOp_1,^cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+^cnn_block_5/conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????OO@2

Identity"
identityIdentity:output:0*?
_input_shapeso
m:???????????::::::::::::::::::::2?
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_3/batch_normalization_3/ReadVariableOp0cnn_block_3/batch_normalization_3/ReadVariableOp2h
2cnn_block_3/batch_normalization_3/ReadVariableOp_12cnn_block_3/batch_normalization_3/ReadVariableOp_12Z
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2X
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2?
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_4/batch_normalization_4/ReadVariableOp0cnn_block_4/batch_normalization_4/ReadVariableOp2h
2cnn_block_4/batch_normalization_4/ReadVariableOp_12cnn_block_4/batch_normalization_4/ReadVariableOp_12Z
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2X
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2?
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_5/batch_normalization_5/ReadVariableOp0cnn_block_5/batch_normalization_5/ReadVariableOp2h
2cnn_block_5/batch_normalization_5/ReadVariableOp_12cnn_block_5/batch_normalization_5/ReadVariableOp_12Z
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2X
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinput_tensor
?!
?
A__inference_model_1_layer_call_and_return_conditional_losses_6737
input_1
res_block_6648
res_block_6650
res_block_6652
res_block_6654
res_block_6656
res_block_6658
res_block_6660
res_block_6662
res_block_6664
res_block_6666
res_block_6668
res_block_6670
res_block_6672
res_block_6674
res_block_6676
res_block_6678
res_block_6680
res_block_6682
res_block_6684
res_block_6686
res_block_1_6689
res_block_1_6691
res_block_1_6693
res_block_1_6695
res_block_1_6697
res_block_1_6699
res_block_1_6701
res_block_1_6703
res_block_1_6705
res_block_1_6707
res_block_1_6709
res_block_1_6711
res_block_1_6713
res_block_1_6715
res_block_1_6717
res_block_1_6719
res_block_1_6721
res_block_1_6723
res_block_1_6725
res_block_1_6727
dense_2_6731
dense_2_6733
identity??dense_2/StatefulPartitionedCall?!res_block/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
!res_block/StatefulPartitionedCallStatefulPartitionedCallinput_1res_block_6648res_block_6650res_block_6652res_block_6654res_block_6656res_block_6658res_block_6660res_block_6662res_block_6664res_block_6666res_block_6668res_block_6670res_block_6672res_block_6674res_block_6676res_block_6678res_block_6680res_block_6682res_block_6684res_block_6686* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OO@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_61822#
!res_block/StatefulPartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCall*res_block/StatefulPartitionedCall:output:0res_block_1_6689res_block_1_6691res_block_1_6693res_block_1_6695res_block_1_6697res_block_1_6699res_block_1_6701res_block_1_6703res_block_1_6705res_block_1_6707res_block_1_6709res_block_1_6711res_block_1_6713res_block_1_6715res_block_1_6717res_block_1_6719res_block_1_6721res_block_1_6723res_block_1_6725res_block_1_6727* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????''?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_64712%
#res_block_1/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall,res_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_66102
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_6731dense_2_6733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_66282!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall"^res_block/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!res_block/StatefulPartitionedCall!res_block/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?u
?
C__inference_res_block_layer_call_and_return_conditional_losses_7841
input_tensor7
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource8
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource=
9cnn_block_3_batch_normalization_3_readvariableop_resource?
;cnn_block_3_batch_normalization_3_readvariableop_1_resourceN
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource8
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource=
9cnn_block_4_batch_normalization_4_readvariableop_resource?
;cnn_block_4_batch_normalization_4_readvariableop_1_resourceN
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource7
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource8
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource=
9cnn_block_5_batch_normalization_5_readvariableop_resource?
;cnn_block_5_batch_normalization_5_readvariableop_1_resourceN
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity??Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_3/batch_normalization_3/ReadVariableOp?2cnn_block_3/batch_normalization_3/ReadVariableOp_1?+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_4/batch_normalization_4/ReadVariableOp?2cnn_block_4/batch_normalization_4/ReadVariableOp_1?+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_5/batch_normalization_5/ReadVariableOp?2cnn_block_5/batch_normalization_5/ReadVariableOp_1?+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?
cnn_block_3/conv2d_3/Conv2DConv2Dinput_tensor2cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
cnn_block_3/conv2d_3/Conv2D?
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_3/conv2d_3/BiasAdd?
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_3/batch_normalization_3/ReadVariableOp?
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_3/batch_normalization_3/ReadVariableOp_1?
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 24
2cnn_block_3/batch_normalization_3/FusedBatchNormV3?
cnn_block_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_3/Relu?
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?
cnn_block_4/conv2d_4/Conv2DConv2Dcnn_block_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
cnn_block_4/conv2d_4/Conv2D?
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_4/conv2d_4/BiasAdd?
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_4/batch_normalization_4/ReadVariableOp?
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_4/batch_normalization_4/ReadVariableOp_1?
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 24
2cnn_block_4/batch_normalization_4/FusedBatchNormV3?
cnn_block_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_4/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_6/BiasAdd?
addAddV2cnn_block_4/Relu:activations:0conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
add?
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?
cnn_block_5/conv2d_5/Conv2DConv2Dadd:z:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
cnn_block_5/conv2d_5/Conv2D?
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
cnn_block_5/conv2d_5/BiasAdd?
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_5/batch_normalization_5/ReadVariableOp?
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_5/batch_normalization_5/ReadVariableOp_1?
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2cnn_block_5/batch_normalization_5/FusedBatchNormV3?
cnn_block_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
cnn_block_5/Relu?
max_pooling2d/MaxPoolMaxPoolcnn_block_5/Relu:activations:0*/
_output_shapes
:?????????OO@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?	
IdentityIdentitymax_pooling2d/MaxPool:output:0B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^cnn_block_3/batch_normalization_3/ReadVariableOp3^cnn_block_3/batch_normalization_3/ReadVariableOp_1,^cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+^cnn_block_3/conv2d_3/Conv2D/ReadVariableOpB^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^cnn_block_4/batch_normalization_4/ReadVariableOp3^cnn_block_4/batch_normalization_4/ReadVariableOp_1,^cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+^cnn_block_4/conv2d_4/Conv2D/ReadVariableOpB^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^cnn_block_5/batch_normalization_5/ReadVariableOp3^cnn_block_5/batch_normalization_5/ReadVariableOp_1,^cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+^cnn_block_5/conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????OO@2

Identity"
identityIdentity:output:0*?
_input_shapeso
m:???????????::::::::::::::::::::2?
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_3/batch_normalization_3/ReadVariableOp0cnn_block_3/batch_normalization_3/ReadVariableOp2h
2cnn_block_3/batch_normalization_3/ReadVariableOp_12cnn_block_3/batch_normalization_3/ReadVariableOp_12Z
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2X
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2?
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_4/batch_normalization_4/ReadVariableOp0cnn_block_4/batch_normalization_4/ReadVariableOp2h
2cnn_block_4/batch_normalization_4/ReadVariableOp_12cnn_block_4/batch_normalization_4/ReadVariableOp_12Z
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2X
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2?
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_5/batch_normalization_5/ReadVariableOp0cnn_block_5/batch_normalization_5/ReadVariableOp2h
2cnn_block_5/batch_normalization_5/ReadVariableOp_12cnn_block_5/batch_normalization_5/ReadVariableOp_12Z
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2X
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinput_tensor
??
?N
 __inference__traced_restore_9290
file_prefix#
assignvariableop_dense_2_kernel#
assignvariableop_1_dense_2_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate<
8assignvariableop_7_res_block_cnn_block_3_conv2d_3_kernel:
6assignvariableop_8_res_block_cnn_block_3_conv2d_3_biasH
Dassignvariableop_9_res_block_cnn_block_3_batch_normalization_3_gammaH
Dassignvariableop_10_res_block_cnn_block_3_batch_normalization_3_beta=
9assignvariableop_11_res_block_cnn_block_4_conv2d_4_kernel;
7assignvariableop_12_res_block_cnn_block_4_conv2d_4_biasI
Eassignvariableop_13_res_block_cnn_block_4_batch_normalization_4_gammaH
Dassignvariableop_14_res_block_cnn_block_4_batch_normalization_4_beta=
9assignvariableop_15_res_block_cnn_block_5_conv2d_5_kernel;
7assignvariableop_16_res_block_cnn_block_5_conv2d_5_biasI
Eassignvariableop_17_res_block_cnn_block_5_batch_normalization_5_gammaH
Dassignvariableop_18_res_block_cnn_block_5_batch_normalization_5_beta1
-assignvariableop_19_res_block_conv2d_6_kernel/
+assignvariableop_20_res_block_conv2d_6_bias?
;assignvariableop_21_res_block_1_cnn_block_6_conv2d_7_kernel=
9assignvariableop_22_res_block_1_cnn_block_6_conv2d_7_biasK
Gassignvariableop_23_res_block_1_cnn_block_6_batch_normalization_6_gammaJ
Fassignvariableop_24_res_block_1_cnn_block_6_batch_normalization_6_beta?
;assignvariableop_25_res_block_1_cnn_block_7_conv2d_8_kernel=
9assignvariableop_26_res_block_1_cnn_block_7_conv2d_8_biasK
Gassignvariableop_27_res_block_1_cnn_block_7_batch_normalization_7_gammaJ
Fassignvariableop_28_res_block_1_cnn_block_7_batch_normalization_7_beta?
;assignvariableop_29_res_block_1_cnn_block_8_conv2d_9_kernel=
9assignvariableop_30_res_block_1_cnn_block_8_conv2d_9_biasK
Gassignvariableop_31_res_block_1_cnn_block_8_batch_normalization_8_gammaJ
Fassignvariableop_32_res_block_1_cnn_block_8_batch_normalization_8_beta4
0assignvariableop_33_res_block_1_conv2d_10_kernel2
.assignvariableop_34_res_block_1_conv2d_10_biasO
Kassignvariableop_35_res_block_cnn_block_3_batch_normalization_3_moving_meanS
Oassignvariableop_36_res_block_cnn_block_3_batch_normalization_3_moving_varianceO
Kassignvariableop_37_res_block_cnn_block_4_batch_normalization_4_moving_meanS
Oassignvariableop_38_res_block_cnn_block_4_batch_normalization_4_moving_varianceO
Kassignvariableop_39_res_block_cnn_block_5_batch_normalization_5_moving_meanS
Oassignvariableop_40_res_block_cnn_block_5_batch_normalization_5_moving_varianceQ
Massignvariableop_41_res_block_1_cnn_block_6_batch_normalization_6_moving_meanU
Qassignvariableop_42_res_block_1_cnn_block_6_batch_normalization_6_moving_varianceQ
Massignvariableop_43_res_block_1_cnn_block_7_batch_normalization_7_moving_meanU
Qassignvariableop_44_res_block_1_cnn_block_7_batch_normalization_7_moving_varianceQ
Massignvariableop_45_res_block_1_cnn_block_8_batch_normalization_8_moving_meanU
Qassignvariableop_46_res_block_1_cnn_block_8_batch_normalization_8_moving_variance
assignvariableop_47_total
assignvariableop_48_count
assignvariableop_49_total_1
assignvariableop_50_count_1-
)assignvariableop_51_adam_dense_2_kernel_m+
'assignvariableop_52_adam_dense_2_bias_mD
@assignvariableop_53_adam_res_block_cnn_block_3_conv2d_3_kernel_mB
>assignvariableop_54_adam_res_block_cnn_block_3_conv2d_3_bias_mP
Lassignvariableop_55_adam_res_block_cnn_block_3_batch_normalization_3_gamma_mO
Kassignvariableop_56_adam_res_block_cnn_block_3_batch_normalization_3_beta_mD
@assignvariableop_57_adam_res_block_cnn_block_4_conv2d_4_kernel_mB
>assignvariableop_58_adam_res_block_cnn_block_4_conv2d_4_bias_mP
Lassignvariableop_59_adam_res_block_cnn_block_4_batch_normalization_4_gamma_mO
Kassignvariableop_60_adam_res_block_cnn_block_4_batch_normalization_4_beta_mD
@assignvariableop_61_adam_res_block_cnn_block_5_conv2d_5_kernel_mB
>assignvariableop_62_adam_res_block_cnn_block_5_conv2d_5_bias_mP
Lassignvariableop_63_adam_res_block_cnn_block_5_batch_normalization_5_gamma_mO
Kassignvariableop_64_adam_res_block_cnn_block_5_batch_normalization_5_beta_m8
4assignvariableop_65_adam_res_block_conv2d_6_kernel_m6
2assignvariableop_66_adam_res_block_conv2d_6_bias_mF
Bassignvariableop_67_adam_res_block_1_cnn_block_6_conv2d_7_kernel_mD
@assignvariableop_68_adam_res_block_1_cnn_block_6_conv2d_7_bias_mR
Nassignvariableop_69_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_mQ
Massignvariableop_70_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_mF
Bassignvariableop_71_adam_res_block_1_cnn_block_7_conv2d_8_kernel_mD
@assignvariableop_72_adam_res_block_1_cnn_block_7_conv2d_8_bias_mR
Nassignvariableop_73_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_mQ
Massignvariableop_74_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_mF
Bassignvariableop_75_adam_res_block_1_cnn_block_8_conv2d_9_kernel_mD
@assignvariableop_76_adam_res_block_1_cnn_block_8_conv2d_9_bias_mR
Nassignvariableop_77_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_mQ
Massignvariableop_78_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_m;
7assignvariableop_79_adam_res_block_1_conv2d_10_kernel_m9
5assignvariableop_80_adam_res_block_1_conv2d_10_bias_m-
)assignvariableop_81_adam_dense_2_kernel_v+
'assignvariableop_82_adam_dense_2_bias_vD
@assignvariableop_83_adam_res_block_cnn_block_3_conv2d_3_kernel_vB
>assignvariableop_84_adam_res_block_cnn_block_3_conv2d_3_bias_vP
Lassignvariableop_85_adam_res_block_cnn_block_3_batch_normalization_3_gamma_vO
Kassignvariableop_86_adam_res_block_cnn_block_3_batch_normalization_3_beta_vD
@assignvariableop_87_adam_res_block_cnn_block_4_conv2d_4_kernel_vB
>assignvariableop_88_adam_res_block_cnn_block_4_conv2d_4_bias_vP
Lassignvariableop_89_adam_res_block_cnn_block_4_batch_normalization_4_gamma_vO
Kassignvariableop_90_adam_res_block_cnn_block_4_batch_normalization_4_beta_vD
@assignvariableop_91_adam_res_block_cnn_block_5_conv2d_5_kernel_vB
>assignvariableop_92_adam_res_block_cnn_block_5_conv2d_5_bias_vP
Lassignvariableop_93_adam_res_block_cnn_block_5_batch_normalization_5_gamma_vO
Kassignvariableop_94_adam_res_block_cnn_block_5_batch_normalization_5_beta_v8
4assignvariableop_95_adam_res_block_conv2d_6_kernel_v6
2assignvariableop_96_adam_res_block_conv2d_6_bias_vF
Bassignvariableop_97_adam_res_block_1_cnn_block_6_conv2d_7_kernel_vD
@assignvariableop_98_adam_res_block_1_cnn_block_6_conv2d_7_bias_vR
Nassignvariableop_99_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_vR
Nassignvariableop_100_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_vG
Cassignvariableop_101_adam_res_block_1_cnn_block_7_conv2d_8_kernel_vE
Aassignvariableop_102_adam_res_block_1_cnn_block_7_conv2d_8_bias_vS
Oassignvariableop_103_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_vR
Nassignvariableop_104_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_vG
Cassignvariableop_105_adam_res_block_1_cnn_block_8_conv2d_9_kernel_vE
Aassignvariableop_106_adam_res_block_1_cnn_block_8_conv2d_9_bias_vS
Oassignvariableop_107_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_vR
Nassignvariableop_108_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_v<
8assignvariableop_109_adam_res_block_1_conv2d_10_kernel_v:
6assignvariableop_110_adam_res_block_1_conv2d_10_bias_v
identity_112??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?8
value?8B?8pB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?
value?B?pB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*~
dtypest
r2p	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp8assignvariableop_7_res_block_cnn_block_3_conv2d_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_res_block_cnn_block_3_conv2d_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpDassignvariableop_9_res_block_cnn_block_3_batch_normalization_3_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpDassignvariableop_10_res_block_cnn_block_3_batch_normalization_3_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_res_block_cnn_block_4_conv2d_4_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_res_block_cnn_block_4_conv2d_4_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpEassignvariableop_13_res_block_cnn_block_4_batch_normalization_4_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpDassignvariableop_14_res_block_cnn_block_4_batch_normalization_4_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp9assignvariableop_15_res_block_cnn_block_5_conv2d_5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp7assignvariableop_16_res_block_cnn_block_5_conv2d_5_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpEassignvariableop_17_res_block_cnn_block_5_batch_normalization_5_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpDassignvariableop_18_res_block_cnn_block_5_batch_normalization_5_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_res_block_conv2d_6_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_res_block_conv2d_6_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp;assignvariableop_21_res_block_1_cnn_block_6_conv2d_7_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp9assignvariableop_22_res_block_1_cnn_block_6_conv2d_7_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpGassignvariableop_23_res_block_1_cnn_block_6_batch_normalization_6_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpFassignvariableop_24_res_block_1_cnn_block_6_batch_normalization_6_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp;assignvariableop_25_res_block_1_cnn_block_7_conv2d_8_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp9assignvariableop_26_res_block_1_cnn_block_7_conv2d_8_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpGassignvariableop_27_res_block_1_cnn_block_7_batch_normalization_7_gammaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpFassignvariableop_28_res_block_1_cnn_block_7_batch_normalization_7_betaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp;assignvariableop_29_res_block_1_cnn_block_8_conv2d_9_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp9assignvariableop_30_res_block_1_cnn_block_8_conv2d_9_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpGassignvariableop_31_res_block_1_cnn_block_8_batch_normalization_8_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpFassignvariableop_32_res_block_1_cnn_block_8_batch_normalization_8_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_res_block_1_conv2d_10_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp.assignvariableop_34_res_block_1_conv2d_10_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpKassignvariableop_35_res_block_cnn_block_3_batch_normalization_3_moving_meanIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpOassignvariableop_36_res_block_cnn_block_3_batch_normalization_3_moving_varianceIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpKassignvariableop_37_res_block_cnn_block_4_batch_normalization_4_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpOassignvariableop_38_res_block_cnn_block_4_batch_normalization_4_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpKassignvariableop_39_res_block_cnn_block_5_batch_normalization_5_moving_meanIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpOassignvariableop_40_res_block_cnn_block_5_batch_normalization_5_moving_varianceIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpMassignvariableop_41_res_block_1_cnn_block_6_batch_normalization_6_moving_meanIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpQassignvariableop_42_res_block_1_cnn_block_6_batch_normalization_6_moving_varianceIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpMassignvariableop_43_res_block_1_cnn_block_7_batch_normalization_7_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpQassignvariableop_44_res_block_1_cnn_block_7_batch_normalization_7_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpMassignvariableop_45_res_block_1_cnn_block_8_batch_normalization_8_moving_meanIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpQassignvariableop_46_res_block_1_cnn_block_8_batch_normalization_8_moving_varianceIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_1Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp@assignvariableop_53_adam_res_block_cnn_block_3_conv2d_3_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp>assignvariableop_54_adam_res_block_cnn_block_3_conv2d_3_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpLassignvariableop_55_adam_res_block_cnn_block_3_batch_normalization_3_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpKassignvariableop_56_adam_res_block_cnn_block_3_batch_normalization_3_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp@assignvariableop_57_adam_res_block_cnn_block_4_conv2d_4_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp>assignvariableop_58_adam_res_block_cnn_block_4_conv2d_4_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpLassignvariableop_59_adam_res_block_cnn_block_4_batch_normalization_4_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpKassignvariableop_60_adam_res_block_cnn_block_4_batch_normalization_4_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp@assignvariableop_61_adam_res_block_cnn_block_5_conv2d_5_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_res_block_cnn_block_5_conv2d_5_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpLassignvariableop_63_adam_res_block_cnn_block_5_batch_normalization_5_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpKassignvariableop_64_adam_res_block_cnn_block_5_batch_normalization_5_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp4assignvariableop_65_adam_res_block_conv2d_6_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp2assignvariableop_66_adam_res_block_conv2d_6_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpBassignvariableop_67_adam_res_block_1_cnn_block_6_conv2d_7_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp@assignvariableop_68_adam_res_block_1_cnn_block_6_conv2d_7_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpNassignvariableop_69_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpMassignvariableop_70_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpBassignvariableop_71_adam_res_block_1_cnn_block_7_conv2d_8_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp@assignvariableop_72_adam_res_block_1_cnn_block_7_conv2d_8_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOpNassignvariableop_73_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOpMassignvariableop_74_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpBassignvariableop_75_adam_res_block_1_cnn_block_8_conv2d_9_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp@assignvariableop_76_adam_res_block_1_cnn_block_8_conv2d_9_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOpNassignvariableop_77_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOpMassignvariableop_78_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_res_block_1_conv2d_10_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp5assignvariableop_80_adam_res_block_1_conv2d_10_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adam_dense_2_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp'assignvariableop_82_adam_dense_2_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp@assignvariableop_83_adam_res_block_cnn_block_3_conv2d_3_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp>assignvariableop_84_adam_res_block_cnn_block_3_conv2d_3_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOpLassignvariableop_85_adam_res_block_cnn_block_3_batch_normalization_3_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOpKassignvariableop_86_adam_res_block_cnn_block_3_batch_normalization_3_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp@assignvariableop_87_adam_res_block_cnn_block_4_conv2d_4_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp>assignvariableop_88_adam_res_block_cnn_block_4_conv2d_4_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOpLassignvariableop_89_adam_res_block_cnn_block_4_batch_normalization_4_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOpKassignvariableop_90_adam_res_block_cnn_block_4_batch_normalization_4_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp@assignvariableop_91_adam_res_block_cnn_block_5_conv2d_5_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp>assignvariableop_92_adam_res_block_cnn_block_5_conv2d_5_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOpLassignvariableop_93_adam_res_block_cnn_block_5_batch_normalization_5_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOpKassignvariableop_94_adam_res_block_cnn_block_5_batch_normalization_5_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp4assignvariableop_95_adam_res_block_conv2d_6_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp2assignvariableop_96_adam_res_block_conv2d_6_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOpBassignvariableop_97_adam_res_block_1_cnn_block_6_conv2d_7_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp@assignvariableop_98_adam_res_block_1_cnn_block_6_conv2d_7_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOpNassignvariableop_99_adam_res_block_1_cnn_block_6_batch_normalization_6_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOpNassignvariableop_100_adam_res_block_1_cnn_block_6_batch_normalization_6_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOpCassignvariableop_101_adam_res_block_1_cnn_block_7_conv2d_8_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOpAassignvariableop_102_adam_res_block_1_cnn_block_7_conv2d_8_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOpOassignvariableop_103_adam_res_block_1_cnn_block_7_batch_normalization_7_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOpNassignvariableop_104_adam_res_block_1_cnn_block_7_batch_normalization_7_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOpCassignvariableop_105_adam_res_block_1_cnn_block_8_conv2d_9_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOpAassignvariableop_106_adam_res_block_1_cnn_block_8_conv2d_9_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOpOassignvariableop_107_adam_res_block_1_cnn_block_8_batch_normalization_8_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOpNassignvariableop_108_adam_res_block_1_cnn_block_8_batch_normalization_8_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp8assignvariableop_109_adam_res_block_1_conv2d_10_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp6assignvariableop_110_adam_res_block_1_conv2d_10_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1109
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_111Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_111?
Identity_112IdentityIdentity_111:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_112"%
identity_112Identity_112:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5791

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_8565

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_res_block_layer_call_fn_7931
input_tensor
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OO@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_61822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OO@2

Identity"
identityIdentity:output:0*?
_input_shapeso
m:???????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinput_tensor
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8227

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_res_block_1_layer_call_fn_8177
input_tensor
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????''?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_64712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????''?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????OO@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????OO@
&
_user_specified_nameinput_tensor
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5895

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_8437

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_8547

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5540

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5692

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8309

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?-
A__inference_model_1_layer_call_and_return_conditional_losses_7507

inputsA
=res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resourceB
>res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resourceG
Cres_block_cnn_block_3_batch_normalization_3_readvariableop_resourceI
Eres_block_cnn_block_3_batch_normalization_3_readvariableop_1_resourceX
Tres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceZ
Vres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceA
=res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resourceB
>res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resourceG
Cres_block_cnn_block_4_batch_normalization_4_readvariableop_resourceI
Eres_block_cnn_block_4_batch_normalization_4_readvariableop_1_resourceX
Tres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceZ
Vres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource5
1res_block_conv2d_6_conv2d_readvariableop_resource6
2res_block_conv2d_6_biasadd_readvariableop_resourceA
=res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resourceB
>res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resourceG
Cres_block_cnn_block_5_batch_normalization_5_readvariableop_resourceI
Eres_block_cnn_block_5_batch_normalization_5_readvariableop_1_resourceX
Tres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceZ
Vres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceC
?res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resourceD
@res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resourceI
Eres_block_1_cnn_block_6_batch_normalization_6_readvariableop_resourceK
Gres_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resourceZ
Vres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource\
Xres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceC
?res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resourceD
@res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resourceI
Eres_block_1_cnn_block_7_batch_normalization_7_readvariableop_resourceK
Gres_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resourceZ
Vres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource\
Xres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource8
4res_block_1_conv2d_10_conv2d_readvariableop_resource9
5res_block_1_conv2d_10_biasadd_readvariableop_resourceC
?res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resourceD
@res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resourceI
Eres_block_1_cnn_block_8_batch_normalization_8_readvariableop_resourceK
Gres_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resourceZ
Vres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource\
Xres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp?<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1?5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp?<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1?5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp?<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1?5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?)res_block/conv2d_6/BiasAdd/ReadVariableOp?(res_block/conv2d_6/Conv2D/ReadVariableOp?Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1?7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1?7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1?7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?,res_block_1/conv2d_10/BiasAdd/ReadVariableOp?+res_block_1/conv2d_10/Conv2D/ReadVariableOp?
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype026
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?
%res_block/cnn_block_3/conv2d_3/Conv2DConv2Dinputs<res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2'
%res_block/cnn_block_3/conv2d_3/Conv2D?
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?
&res_block/cnn_block_3/conv2d_3/BiasAddBiasAdd.res_block/cnn_block_3/conv2d_3/Conv2D:output:0=res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2(
&res_block/cnn_block_3/conv2d_3/BiasAdd?
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpCres_block_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02<
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp?
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02>
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1?
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02M
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02O
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
<res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_3/conv2d_3/BiasAdd:output:0Bres_block/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Dres_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Sres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2>
<res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3?
res_block/cnn_block_3/ReluRelu@res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
res_block/cnn_block_3/Relu?
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype026
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?
%res_block/cnn_block_4/conv2d_4/Conv2DConv2D(res_block/cnn_block_3/Relu:activations:0<res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2'
%res_block/cnn_block_4/conv2d_4/Conv2D?
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?
&res_block/cnn_block_4/conv2d_4/BiasAddBiasAdd.res_block/cnn_block_4/conv2d_4/Conv2D:output:0=res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2(
&res_block/cnn_block_4/conv2d_4/BiasAdd?
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpCres_block_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02<
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp?
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02>
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1?
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02M
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02O
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
<res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_4/conv2d_4/BiasAdd:output:0Bres_block/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Dres_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Sres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2>
<res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3?
res_block/cnn_block_4/ReluRelu@res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
res_block/cnn_block_4/Relu?
(res_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp1res_block_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(res_block/conv2d_6/Conv2D/ReadVariableOp?
res_block/conv2d_6/Conv2DConv2Dinputs0res_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
res_block/conv2d_6/Conv2D?
)res_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2res_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)res_block/conv2d_6/BiasAdd/ReadVariableOp?
res_block/conv2d_6/BiasAddBiasAdd"res_block/conv2d_6/Conv2D:output:01res_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
res_block/conv2d_6/BiasAdd?
res_block/addAddV2(res_block/cnn_block_4/Relu:activations:0#res_block/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
res_block/add?
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype026
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?
%res_block/cnn_block_5/conv2d_5/Conv2DConv2Dres_block/add:z:0<res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2'
%res_block/cnn_block_5/conv2d_5/Conv2D?
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?
&res_block/cnn_block_5/conv2d_5/BiasAddBiasAdd.res_block/cnn_block_5/conv2d_5/Conv2D:output:0=res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2(
&res_block/cnn_block_5/conv2d_5/BiasAdd?
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpCres_block_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02<
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp?
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1?
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02M
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02O
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
<res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_5/conv2d_5/BiasAdd:output:0Bres_block/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Dres_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Sres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2>
<res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3?
res_block/cnn_block_5/ReluRelu@res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
res_block/cnn_block_5/Relu?
res_block/max_pooling2d/MaxPoolMaxPool(res_block/cnn_block_5/Relu:activations:0*/
_output_shapes
:?????????OO@*
ksize
*
paddingVALID*
strides
2!
res_block/max_pooling2d/MaxPool?
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype028
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?
'res_block_1/cnn_block_6/conv2d_7/Conv2DConv2D(res_block/max_pooling2d/MaxPool:output:0>res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2)
'res_block_1/cnn_block_6/conv2d_7/Conv2D?
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?
(res_block_1/cnn_block_6/conv2d_7/BiasAddBiasAdd0res_block_1/cnn_block_6/conv2d_7/Conv2D:output:0?res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2*
(res_block_1/cnn_block_6/conv2d_7/BiasAdd?
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpEres_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02@
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1?
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Q
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
>res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_6/conv2d_7/BiasAdd:output:0Dres_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Fres_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Ures_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2@
>res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3?
res_block_1/cnn_block_6/ReluReluBres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/cnn_block_6/Relu?
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype028
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?
'res_block_1/cnn_block_7/conv2d_8/Conv2DConv2D*res_block_1/cnn_block_6/Relu:activations:0>res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2)
'res_block_1/cnn_block_7/conv2d_8/Conv2D?
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?
(res_block_1/cnn_block_7/conv2d_8/BiasAddBiasAdd0res_block_1/cnn_block_7/conv2d_8/Conv2D:output:0?res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2*
(res_block_1/cnn_block_7/conv2d_8/BiasAdd?
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOpEres_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02@
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1?
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Q
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
>res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_7/conv2d_8/BiasAdd:output:0Dres_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp:value:0Fres_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Ures_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2@
>res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3?
res_block_1/cnn_block_7/ReluReluBres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/cnn_block_7/Relu?
+res_block_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4res_block_1_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+res_block_1/conv2d_10/Conv2D/ReadVariableOp?
res_block_1/conv2d_10/Conv2DConv2D(res_block/max_pooling2d/MaxPool:output:03res_block_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
res_block_1/conv2d_10/Conv2D?
,res_block_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5res_block_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,res_block_1/conv2d_10/BiasAdd/ReadVariableOp?
res_block_1/conv2d_10/BiasAddBiasAdd%res_block_1/conv2d_10/Conv2D:output:04res_block_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/conv2d_10/BiasAdd?
res_block_1/addAddV2*res_block_1/cnn_block_7/Relu:activations:0&res_block_1/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/add?
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype028
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?
'res_block_1/cnn_block_8/conv2d_9/Conv2DConv2Dres_block_1/add:z:0>res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2)
'res_block_1/cnn_block_8/conv2d_9/Conv2D?
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?
(res_block_1/cnn_block_8/conv2d_9/BiasAddBiasAdd0res_block_1/cnn_block_8/conv2d_9/Conv2D:output:0?res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2*
(res_block_1/cnn_block_8/conv2d_9/BiasAdd?
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOpEres_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02@
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1?
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Q
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
>res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_8/conv2d_9/BiasAdd:output:0Dres_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp:value:0Fres_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Ures_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2@
>res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3?
res_block_1/cnn_block_8/ReluReluBres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/cnn_block_8/Relu?
#res_block_1/max_pooling2d_1/MaxPoolMaxPool*res_block_1/cnn_block_8/Relu:activations:0*0
_output_shapes
:?????????''?*
ksize
*
paddingVALID*
strides
2%
#res_block_1/max_pooling2d_1/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
flatten_1/Const?
flatten_1/ReshapeReshape,res_block_1/max_pooling2d_1/MaxPool:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_1/Reshape?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_2/BiasAdd?
IdentityIdentitydense_2/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOpL^res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_3/batch_normalization_3/ReadVariableOp=^res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_16^res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp5^res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpL^res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_4/batch_normalization_4/ReadVariableOp=^res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_16^res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp5^res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpL^res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_5/batch_normalization_5/ReadVariableOp=^res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_16^res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp5^res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*^res_block/conv2d_6/BiasAdd/ReadVariableOp)^res_block/conv2d_6/Conv2D/ReadVariableOpN^res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?^res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_18^res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpN^res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?^res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_18^res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpN^res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?^res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_18^res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp-^res_block_1/conv2d_10/BiasAdd/ReadVariableOp,^res_block_1/conv2d_10/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2?
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp2|
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_12n
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2?
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp2|
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_12n
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2?
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp2|
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_12n
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2V
)res_block/conv2d_6/BiasAdd/ReadVariableOp)res_block/conv2d_6/BiasAdd/ReadVariableOp2T
(res_block/conv2d_6/Conv2D/ReadVariableOp(res_block/conv2d_6/Conv2D/ReadVariableOp2?
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp2?
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_12r
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2?
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp2?
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_12r
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2?
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp2?
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_12r
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2\
,res_block_1/conv2d_10/BiasAdd/ReadVariableOp,res_block_1/conv2d_10/BiasAdd/ReadVariableOp2Z
+res_block_1/conv2d_10/Conv2D/ReadVariableOp+res_block_1/conv2d_10/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
ۗ
?
E__inference_res_block_1_layer_call_and_return_conditional_losses_8012
input_tensor7
3cnn_block_6_conv2d_7_conv2d_readvariableop_resource8
4cnn_block_6_conv2d_7_biasadd_readvariableop_resource=
9cnn_block_6_batch_normalization_6_readvariableop_resource?
;cnn_block_6_batch_normalization_6_readvariableop_1_resourceN
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_7_conv2d_8_conv2d_readvariableop_resource8
4cnn_block_7_conv2d_8_biasadd_readvariableop_resource=
9cnn_block_7_batch_normalization_7_readvariableop_resource?
;cnn_block_7_batch_normalization_7_readvariableop_1_resourceN
Jcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource7
3cnn_block_8_conv2d_9_conv2d_readvariableop_resource8
4cnn_block_8_conv2d_9_biasadd_readvariableop_resource=
9cnn_block_8_batch_normalization_8_readvariableop_resource?
;cnn_block_8_batch_normalization_8_readvariableop_1_resourceN
Jcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??0cnn_block_6/batch_normalization_6/AssignNewValue?2cnn_block_6/batch_normalization_6/AssignNewValue_1?Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_6/batch_normalization_6/ReadVariableOp?2cnn_block_6/batch_normalization_6/ReadVariableOp_1?+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?0cnn_block_7/batch_normalization_7/AssignNewValue?2cnn_block_7/batch_normalization_7/AssignNewValue_1?Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_7/batch_normalization_7/ReadVariableOp?2cnn_block_7/batch_normalization_7/ReadVariableOp_1?+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?0cnn_block_8/batch_normalization_8/AssignNewValue?2cnn_block_8/batch_normalization_8/AssignNewValue_1?Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_8/batch_normalization_8/ReadVariableOp?2cnn_block_8/batch_normalization_8/ReadVariableOp_1?+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?
cnn_block_6/conv2d_7/Conv2DConv2Dinput_tensor2cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_6/conv2d_7/Conv2D?
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?
cnn_block_6/conv2d_7/BiasAddBiasAdd$cnn_block_6/conv2d_7/Conv2D:output:03cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_6/conv2d_7/BiasAdd?
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_6/batch_normalization_6/ReadVariableOp?
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_6/batch_normalization_6/ReadVariableOp_1?
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_7/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_6/batch_normalization_6/FusedBatchNormV3?
0cnn_block_6/batch_normalization_6/AssignNewValueAssignVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource?cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_6/batch_normalization_6/AssignNewValue?
2cnn_block_6/batch_normalization_6/AssignNewValue_1AssignVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0D^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_6/batch_normalization_6/AssignNewValue_1?
cnn_block_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_6/Relu?
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?
cnn_block_7/conv2d_8/Conv2DConv2Dcnn_block_6/Relu:activations:02cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_7/conv2d_8/Conv2D?
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?
cnn_block_7/conv2d_8/BiasAddBiasAdd$cnn_block_7/conv2d_8/Conv2D:output:03cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_7/conv2d_8/BiasAdd?
0cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_7/batch_normalization_7/ReadVariableOp?
2cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_7/batch_normalization_7/ReadVariableOp_1?
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_7/conv2d_8/BiasAdd:output:08cnn_block_7/batch_normalization_7/ReadVariableOp:value:0:cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_7/batch_normalization_7/FusedBatchNormV3?
0cnn_block_7/batch_normalization_7/AssignNewValueAssignVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource?cnn_block_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0B^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_7/batch_normalization_7/AssignNewValue?
2cnn_block_7/batch_normalization_7/AssignNewValue_1AssignVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0D^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_7/batch_normalization_7/AssignNewValue_1?
cnn_block_7/ReluRelu6cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_7/Relu?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinput_tensor'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
conv2d_10/BiasAdd?
addAddV2cnn_block_7/Relu:activations:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????OO?2
add?
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?
cnn_block_8/conv2d_9/Conv2DConv2Dadd:z:02cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_8/conv2d_9/Conv2D?
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?
cnn_block_8/conv2d_9/BiasAddBiasAdd$cnn_block_8/conv2d_9/Conv2D:output:03cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_8/conv2d_9/BiasAdd?
0cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_8/batch_normalization_8/ReadVariableOp?
2cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_8/batch_normalization_8/ReadVariableOp_1?
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_8/conv2d_9/BiasAdd:output:08cnn_block_8/batch_normalization_8/ReadVariableOp:value:0:cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_8/batch_normalization_8/FusedBatchNormV3?
0cnn_block_8/batch_normalization_8/AssignNewValueAssignVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource?cnn_block_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0B^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_8/batch_normalization_8/AssignNewValue?
2cnn_block_8/batch_normalization_8/AssignNewValue_1AssignVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0D^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_8/batch_normalization_8/AssignNewValue_1?
cnn_block_8/ReluRelu6cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_8/Relu?
max_pooling2d_1/MaxPoolMaxPoolcnn_block_8/Relu:activations:0*0
_output_shapes
:?????????''?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
IdentityIdentity max_pooling2d_1/MaxPool:output:01^cnn_block_6/batch_normalization_6/AssignNewValue3^cnn_block_6/batch_normalization_6/AssignNewValue_1B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^cnn_block_6/batch_normalization_6/ReadVariableOp3^cnn_block_6/batch_normalization_6/ReadVariableOp_1,^cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+^cnn_block_6/conv2d_7/Conv2D/ReadVariableOp1^cnn_block_7/batch_normalization_7/AssignNewValue3^cnn_block_7/batch_normalization_7/AssignNewValue_1B^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^cnn_block_7/batch_normalization_7/ReadVariableOp3^cnn_block_7/batch_normalization_7/ReadVariableOp_1,^cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+^cnn_block_7/conv2d_8/Conv2D/ReadVariableOp1^cnn_block_8/batch_normalization_8/AssignNewValue3^cnn_block_8/batch_normalization_8/AssignNewValue_1B^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^cnn_block_8/batch_normalization_8/ReadVariableOp3^cnn_block_8/batch_normalization_8/ReadVariableOp_1,^cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+^cnn_block_8/conv2d_9/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????''?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????OO@::::::::::::::::::::2d
0cnn_block_6/batch_normalization_6/AssignNewValue0cnn_block_6/batch_normalization_6/AssignNewValue2h
2cnn_block_6/batch_normalization_6/AssignNewValue_12cnn_block_6/batch_normalization_6/AssignNewValue_12?
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_6/batch_normalization_6/ReadVariableOp0cnn_block_6/batch_normalization_6/ReadVariableOp2h
2cnn_block_6/batch_normalization_6/ReadVariableOp_12cnn_block_6/batch_normalization_6/ReadVariableOp_12Z
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2X
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2d
0cnn_block_7/batch_normalization_7/AssignNewValue0cnn_block_7/batch_normalization_7/AssignNewValue2h
2cnn_block_7/batch_normalization_7/AssignNewValue_12cnn_block_7/batch_normalization_7/AssignNewValue_12?
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_7/batch_normalization_7/ReadVariableOp0cnn_block_7/batch_normalization_7/ReadVariableOp2h
2cnn_block_7/batch_normalization_7/ReadVariableOp_12cnn_block_7/batch_normalization_7/ReadVariableOp_12Z
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2X
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2d
0cnn_block_8/batch_normalization_8/AssignNewValue0cnn_block_8/batch_normalization_8/AssignNewValue2h
2cnn_block_8/batch_normalization_8/AssignNewValue_12cnn_block_8/batch_normalization_8/AssignNewValue_12?
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpAcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_8/batch_normalization_8/ReadVariableOp0cnn_block_8/batch_normalization_8/ReadVariableOp2h
2cnn_block_8/batch_normalization_8/ReadVariableOp_12cnn_block_8/batch_normalization_8/ReadVariableOp_12Z
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2X
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:?????????OO@
&
_user_specified_nameinput_tensor
?
?
&__inference_model_1_layer_call_fn_6919
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_68322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
*__inference_res_block_1_layer_call_fn_8132
input_tensor
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????''?*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_63962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????''?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????OO@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????OO@
&
_user_specified_nameinput_tensor
?
?
4__inference_batch_normalization_4_layer_call_fn_8335

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_55712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_7_layer_call_fn_8527

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_58952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_6628

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_8198

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6016

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5436

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_8183

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????''?:X T
0
_output_shapes
:?????????''?
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_1_layer_call_fn_6022

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_60162
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5571

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_8_layer_call_fn_8591

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_59992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
A__inference_model_1_layer_call_and_return_conditional_losses_6645
input_1
res_block_6274
res_block_6276
res_block_6278
res_block_6280
res_block_6282
res_block_6284
res_block_6286
res_block_6288
res_block_6290
res_block_6292
res_block_6294
res_block_6296
res_block_6298
res_block_6300
res_block_6302
res_block_6304
res_block_6306
res_block_6308
res_block_6310
res_block_6312
res_block_1_6563
res_block_1_6565
res_block_1_6567
res_block_1_6569
res_block_1_6571
res_block_1_6573
res_block_1_6575
res_block_1_6577
res_block_1_6579
res_block_1_6581
res_block_1_6583
res_block_1_6585
res_block_1_6587
res_block_1_6589
res_block_1_6591
res_block_1_6593
res_block_1_6595
res_block_1_6597
res_block_1_6599
res_block_1_6601
dense_2_6639
dense_2_6641
identity??dense_2/StatefulPartitionedCall?!res_block/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
!res_block/StatefulPartitionedCallStatefulPartitionedCallinput_1res_block_6274res_block_6276res_block_6278res_block_6280res_block_6282res_block_6284res_block_6286res_block_6288res_block_6290res_block_6292res_block_6294res_block_6296res_block_6298res_block_6300res_block_6302res_block_6304res_block_6306res_block_6308res_block_6310res_block_6312* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OO@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_61822#
!res_block/StatefulPartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCall*res_block/StatefulPartitionedCall:output:0res_block_1_6563res_block_1_6565res_block_1_6567res_block_1_6569res_block_1_6571res_block_1_6573res_block_1_6575res_block_1_6577res_block_1_6579res_block_1_6581res_block_1_6583res_block_1_6585res_block_1_6587res_block_1_6589res_block_1_6591res_block_1_6593res_block_1_6595res_block_1_6597res_block_1_6599res_block_1_6601* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????''?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_64712%
#res_block_1/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall,res_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_66102
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_6639dense_2_6641*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_66282!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall"^res_block/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!res_block/StatefulPartitionedCall!res_block/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?!
?
A__inference_model_1_layer_call_and_return_conditional_losses_7013

inputs
res_block_6924
res_block_6926
res_block_6928
res_block_6930
res_block_6932
res_block_6934
res_block_6936
res_block_6938
res_block_6940
res_block_6942
res_block_6944
res_block_6946
res_block_6948
res_block_6950
res_block_6952
res_block_6954
res_block_6956
res_block_6958
res_block_6960
res_block_6962
res_block_1_6965
res_block_1_6967
res_block_1_6969
res_block_1_6971
res_block_1_6973
res_block_1_6975
res_block_1_6977
res_block_1_6979
res_block_1_6981
res_block_1_6983
res_block_1_6985
res_block_1_6987
res_block_1_6989
res_block_1_6991
res_block_1_6993
res_block_1_6995
res_block_1_6997
res_block_1_6999
res_block_1_7001
res_block_1_7003
dense_2_7007
dense_2_7009
identity??dense_2/StatefulPartitionedCall?!res_block/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
!res_block/StatefulPartitionedCallStatefulPartitionedCallinputsres_block_6924res_block_6926res_block_6928res_block_6930res_block_6932res_block_6934res_block_6936res_block_6938res_block_6940res_block_6942res_block_6944res_block_6946res_block_6948res_block_6950res_block_6952res_block_6954res_block_6956res_block_6958res_block_6960res_block_6962* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OO@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_61822#
!res_block/StatefulPartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCall*res_block/StatefulPartitionedCall:output:0res_block_1_6965res_block_1_6967res_block_1_6969res_block_1_6971res_block_1_6973res_block_1_6975res_block_1_6977res_block_1_6979res_block_1_6981res_block_1_6983res_block_1_6985res_block_1_6987res_block_1_6989res_block_1_6991res_block_1_6993res_block_1_6995res_block_1_6997res_block_1_6999res_block_1_7001res_block_1_7003* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????''?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_64712%
#res_block_1/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall,res_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_66102
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_7007dense_2_7009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_66282!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall"^res_block/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!res_block/StatefulPartitionedCall!res_block/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_model_1_layer_call_fn_7596

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_68322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_8419

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
A__inference_model_1_layer_call_and_return_conditional_losses_6832

inputs
res_block_6743
res_block_6745
res_block_6747
res_block_6749
res_block_6751
res_block_6753
res_block_6755
res_block_6757
res_block_6759
res_block_6761
res_block_6763
res_block_6765
res_block_6767
res_block_6769
res_block_6771
res_block_6773
res_block_6775
res_block_6777
res_block_6779
res_block_6781
res_block_1_6784
res_block_1_6786
res_block_1_6788
res_block_1_6790
res_block_1_6792
res_block_1_6794
res_block_1_6796
res_block_1_6798
res_block_1_6800
res_block_1_6802
res_block_1_6804
res_block_1_6806
res_block_1_6808
res_block_1_6810
res_block_1_6812
res_block_1_6814
res_block_1_6816
res_block_1_6818
res_block_1_6820
res_block_1_6822
dense_2_6826
dense_2_6828
identity??dense_2/StatefulPartitionedCall?!res_block/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
!res_block/StatefulPartitionedCallStatefulPartitionedCallinputsres_block_6743res_block_6745res_block_6747res_block_6749res_block_6751res_block_6753res_block_6755res_block_6757res_block_6759res_block_6761res_block_6763res_block_6765res_block_6767res_block_6769res_block_6771res_block_6773res_block_6775res_block_6777res_block_6779res_block_6781* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OO@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_61822#
!res_block/StatefulPartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCall*res_block/StatefulPartitionedCall:output:0res_block_1_6784res_block_1_6786res_block_1_6788res_block_1_6790res_block_1_6792res_block_1_6794res_block_1_6796res_block_1_6798res_block_1_6800res_block_1_6802res_block_1_6804res_block_1_6806res_block_1_6808res_block_1_6810res_block_1_6812res_block_1_6814res_block_1_6816res_block_1_6818res_block_1_6820res_block_1_6822* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????''?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_res_block_1_layer_call_and_return_conditional_losses_64712%
#res_block_1/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall,res_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_66102
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_6826dense_2_6828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_66282!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall"^res_block/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!res_block/StatefulPartitionedCall!res_block/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_7199
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_53742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
??
?-
A__inference_model_1_layer_call_and_return_conditional_losses_7353

inputsA
=res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resourceB
>res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resourceG
Cres_block_cnn_block_3_batch_normalization_3_readvariableop_resourceI
Eres_block_cnn_block_3_batch_normalization_3_readvariableop_1_resourceX
Tres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceZ
Vres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceA
=res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resourceB
>res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resourceG
Cres_block_cnn_block_4_batch_normalization_4_readvariableop_resourceI
Eres_block_cnn_block_4_batch_normalization_4_readvariableop_1_resourceX
Tres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceZ
Vres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource5
1res_block_conv2d_6_conv2d_readvariableop_resource6
2res_block_conv2d_6_biasadd_readvariableop_resourceA
=res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resourceB
>res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resourceG
Cres_block_cnn_block_5_batch_normalization_5_readvariableop_resourceI
Eres_block_cnn_block_5_batch_normalization_5_readvariableop_1_resourceX
Tres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceZ
Vres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceC
?res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resourceD
@res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resourceI
Eres_block_1_cnn_block_6_batch_normalization_6_readvariableop_resourceK
Gres_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resourceZ
Vres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource\
Xres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceC
?res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resourceD
@res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resourceI
Eres_block_1_cnn_block_7_batch_normalization_7_readvariableop_resourceK
Gres_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resourceZ
Vres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource\
Xres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource8
4res_block_1_conv2d_10_conv2d_readvariableop_resource9
5res_block_1_conv2d_10_biasadd_readvariableop_resourceC
?res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resourceD
@res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resourceI
Eres_block_1_cnn_block_8_batch_normalization_8_readvariableop_resourceK
Gres_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resourceZ
Vres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource\
Xres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp?<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1?5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp?<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1?5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp?<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1?5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?)res_block/conv2d_6/BiasAdd/ReadVariableOp?(res_block/conv2d_6/Conv2D/ReadVariableOp?Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1?7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1?7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1?7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?,res_block_1/conv2d_10/BiasAdd/ReadVariableOp?+res_block_1/conv2d_10/Conv2D/ReadVariableOp?
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype026
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?
%res_block/cnn_block_3/conv2d_3/Conv2DConv2Dinputs<res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2'
%res_block/cnn_block_3/conv2d_3/Conv2D?
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?
&res_block/cnn_block_3/conv2d_3/BiasAddBiasAdd.res_block/cnn_block_3/conv2d_3/Conv2D:output:0=res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2(
&res_block/cnn_block_3/conv2d_3/BiasAdd?
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpCres_block_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02<
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp?
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02>
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1?
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02M
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02O
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
<res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_3/conv2d_3/BiasAdd:output:0Bres_block/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Dres_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Sres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2>
<res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3?
res_block/cnn_block_3/ReluRelu@res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
res_block/cnn_block_3/Relu?
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype026
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?
%res_block/cnn_block_4/conv2d_4/Conv2DConv2D(res_block/cnn_block_3/Relu:activations:0<res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2'
%res_block/cnn_block_4/conv2d_4/Conv2D?
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?
&res_block/cnn_block_4/conv2d_4/BiasAddBiasAdd.res_block/cnn_block_4/conv2d_4/Conv2D:output:0=res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2(
&res_block/cnn_block_4/conv2d_4/BiasAdd?
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpCres_block_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype02<
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp?
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype02>
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1?
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02M
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02O
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
<res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_4/conv2d_4/BiasAdd:output:0Bres_block/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Dres_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Sres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2>
<res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3?
res_block/cnn_block_4/ReluRelu@res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
res_block/cnn_block_4/Relu?
(res_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp1res_block_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(res_block/conv2d_6/Conv2D/ReadVariableOp?
res_block/conv2d_6/Conv2DConv2Dinputs0res_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
res_block/conv2d_6/Conv2D?
)res_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2res_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)res_block/conv2d_6/BiasAdd/ReadVariableOp?
res_block/conv2d_6/BiasAddBiasAdd"res_block/conv2d_6/Conv2D:output:01res_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
res_block/conv2d_6/BiasAdd?
res_block/addAddV2(res_block/cnn_block_4/Relu:activations:0#res_block/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
res_block/add?
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp=res_block_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype026
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?
%res_block/cnn_block_5/conv2d_5/Conv2DConv2Dres_block/add:z:0<res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2'
%res_block/cnn_block_5/conv2d_5/Conv2D?
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp>res_block_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?
&res_block/cnn_block_5/conv2d_5/BiasAddBiasAdd.res_block/cnn_block_5/conv2d_5/Conv2D:output:0=res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2(
&res_block/cnn_block_5/conv2d_5/BiasAdd?
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpCres_block_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02<
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp?
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpEres_block_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1?
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpTres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02M
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVres_block_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02O
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
<res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3/res_block/cnn_block_5/conv2d_5/BiasAdd:output:0Bres_block/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Dres_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Sres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Ures_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2>
<res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3?
res_block/cnn_block_5/ReluRelu@res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
res_block/cnn_block_5/Relu?
res_block/max_pooling2d/MaxPoolMaxPool(res_block/cnn_block_5/Relu:activations:0*/
_output_shapes
:?????????OO@*
ksize
*
paddingVALID*
strides
2!
res_block/max_pooling2d/MaxPool?
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype028
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?
'res_block_1/cnn_block_6/conv2d_7/Conv2DConv2D(res_block/max_pooling2d/MaxPool:output:0>res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2)
'res_block_1/cnn_block_6/conv2d_7/Conv2D?
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?
(res_block_1/cnn_block_6/conv2d_7/BiasAddBiasAdd0res_block_1/cnn_block_6/conv2d_7/Conv2D:output:0?res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2*
(res_block_1/cnn_block_6/conv2d_7/BiasAdd?
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpEres_block_1_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02@
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1?
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Q
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
>res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_6/conv2d_7/BiasAdd:output:0Dres_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Fres_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Ures_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2@
>res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3?
res_block_1/cnn_block_6/ReluReluBres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/cnn_block_6/Relu?
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype028
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?
'res_block_1/cnn_block_7/conv2d_8/Conv2DConv2D*res_block_1/cnn_block_6/Relu:activations:0>res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2)
'res_block_1/cnn_block_7/conv2d_8/Conv2D?
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?
(res_block_1/cnn_block_7/conv2d_8/BiasAddBiasAdd0res_block_1/cnn_block_7/conv2d_8/Conv2D:output:0?res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2*
(res_block_1/cnn_block_7/conv2d_8/BiasAdd?
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOpEres_block_1_cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02@
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1?
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Q
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
>res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_7/conv2d_8/BiasAdd:output:0Dres_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp:value:0Fres_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Ures_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2@
>res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3?
res_block_1/cnn_block_7/ReluReluBres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/cnn_block_7/Relu?
+res_block_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4res_block_1_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+res_block_1/conv2d_10/Conv2D/ReadVariableOp?
res_block_1/conv2d_10/Conv2DConv2D(res_block/max_pooling2d/MaxPool:output:03res_block_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
res_block_1/conv2d_10/Conv2D?
,res_block_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5res_block_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,res_block_1/conv2d_10/BiasAdd/ReadVariableOp?
res_block_1/conv2d_10/BiasAddBiasAdd%res_block_1/conv2d_10/Conv2D:output:04res_block_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/conv2d_10/BiasAdd?
res_block_1/addAddV2*res_block_1/cnn_block_7/Relu:activations:0&res_block_1/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/add?
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp?res_block_1_cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype028
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?
'res_block_1/cnn_block_8/conv2d_9/Conv2DConv2Dres_block_1/add:z:0>res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2)
'res_block_1/cnn_block_8/conv2d_9/Conv2D?
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp@res_block_1_cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?
(res_block_1/cnn_block_8/conv2d_9/BiasAddBiasAdd0res_block_1/cnn_block_8/conv2d_9/Conv2D:output:0?res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2*
(res_block_1/cnn_block_8/conv2d_9/BiasAdd?
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOpEres_block_1_cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOpGres_block_1_cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02@
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1?
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpVres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02O
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXres_block_1_cnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02Q
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
>res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV31res_block_1/cnn_block_8/conv2d_9/BiasAdd:output:0Dres_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp:value:0Fres_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Ures_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Wres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 2@
>res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3?
res_block_1/cnn_block_8/ReluReluBres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
res_block_1/cnn_block_8/Relu?
#res_block_1/max_pooling2d_1/MaxPoolMaxPool*res_block_1/cnn_block_8/Relu:activations:0*0
_output_shapes
:?????????''?*
ksize
*
paddingVALID*
strides
2%
#res_block_1/max_pooling2d_1/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
flatten_1/Const?
flatten_1/ReshapeReshape,res_block_1/max_pooling2d_1/MaxPool:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_1/Reshape?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??
*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_2/BiasAdd?
IdentityIdentitydense_2/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOpL^res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_3/batch_normalization_3/ReadVariableOp=^res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_16^res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp5^res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpL^res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_4/batch_normalization_4/ReadVariableOp=^res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_16^res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp5^res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpL^res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpN^res_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1;^res_block/cnn_block_5/batch_normalization_5/ReadVariableOp=^res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_16^res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp5^res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*^res_block/conv2d_6/BiasAdd/ReadVariableOp)^res_block/conv2d_6/Conv2D/ReadVariableOpN^res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp?^res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_18^res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOpN^res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp?^res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_18^res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOpN^res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpP^res_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=^res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp?^res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_18^res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp7^res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp-^res_block_1/conv2d_10/BiasAdd/ReadVariableOp,^res_block_1/conv2d_10/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2?
Kres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp:res_block/cnn_block_3/batch_normalization_3/ReadVariableOp2|
<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_1<res_block/cnn_block_3/batch_normalization_3/ReadVariableOp_12n
5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp5res_block/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp4res_block/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2?
Kres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp:res_block/cnn_block_4/batch_normalization_4/ReadVariableOp2|
<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_1<res_block/cnn_block_4/batch_normalization_4/ReadVariableOp_12n
5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp5res_block/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp4res_block/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2?
Kres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpKres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Mres_block/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12x
:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp:res_block/cnn_block_5/batch_normalization_5/ReadVariableOp2|
<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_1<res_block/cnn_block_5/batch_normalization_5/ReadVariableOp_12n
5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp5res_block/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2l
4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp4res_block/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2V
)res_block/conv2d_6/BiasAdd/ReadVariableOp)res_block/conv2d_6/BiasAdd/ReadVariableOp2T
(res_block/conv2d_6/Conv2D/ReadVariableOp(res_block/conv2d_6/Conv2D/ReadVariableOp2?
Mres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp<res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp2?
>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_1>res_block_1/cnn_block_6/batch_normalization_6/ReadVariableOp_12r
7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp7res_block_1/cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp6res_block_1/cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2?
Mres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp<res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp2?
>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_1>res_block_1/cnn_block_7/batch_normalization_7/ReadVariableOp_12r
7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp7res_block_1/cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp6res_block_1/cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2?
Mres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpMres_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ores_block_1/cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12|
<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp<res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp2?
>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_1>res_block_1/cnn_block_8/batch_normalization_8/ReadVariableOp_12r
7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp7res_block_1/cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2p
6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp6res_block_1/cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2\
,res_block_1/conv2d_10/BiasAdd/ReadVariableOp,res_block_1/conv2d_10/BiasAdd/ReadVariableOp2Z
+res_block_1/conv2d_10/Conv2D/ReadVariableOp+res_block_1/conv2d_10/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?u
?
E__inference_res_block_1_layer_call_and_return_conditional_losses_6471
input_tensor7
3cnn_block_6_conv2d_7_conv2d_readvariableop_resource8
4cnn_block_6_conv2d_7_biasadd_readvariableop_resource=
9cnn_block_6_batch_normalization_6_readvariableop_resource?
;cnn_block_6_batch_normalization_6_readvariableop_1_resourceN
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_7_conv2d_8_conv2d_readvariableop_resource8
4cnn_block_7_conv2d_8_biasadd_readvariableop_resource=
9cnn_block_7_batch_normalization_7_readvariableop_resource?
;cnn_block_7_batch_normalization_7_readvariableop_1_resourceN
Jcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource7
3cnn_block_8_conv2d_9_conv2d_readvariableop_resource8
4cnn_block_8_conv2d_9_biasadd_readvariableop_resource=
9cnn_block_8_batch_normalization_8_readvariableop_resource?
;cnn_block_8_batch_normalization_8_readvariableop_1_resourceN
Jcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_6/batch_normalization_6/ReadVariableOp?2cnn_block_6/batch_normalization_6/ReadVariableOp_1?+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_7/batch_normalization_7/ReadVariableOp?2cnn_block_7/batch_normalization_7/ReadVariableOp_1?+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_8/batch_normalization_8/ReadVariableOp?2cnn_block_8/batch_normalization_8/ReadVariableOp_1?+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02,
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp?
cnn_block_6/conv2d_7/Conv2DConv2Dinput_tensor2cnn_block_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_6/conv2d_7/Conv2D?
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp?
cnn_block_6/conv2d_7/BiasAddBiasAdd$cnn_block_6/conv2d_7/Conv2D:output:03cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_6/conv2d_7/BiasAdd?
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_6/batch_normalization_6/ReadVariableOp?
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_6/batch_normalization_6/ReadVariableOp_1?
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_7/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 24
2cnn_block_6/batch_normalization_6/FusedBatchNormV3?
cnn_block_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_6/Relu?
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_7_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp?
cnn_block_7/conv2d_8/Conv2DConv2Dcnn_block_6/Relu:activations:02cnn_block_7/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_7/conv2d_8/Conv2D?
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_7_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp?
cnn_block_7/conv2d_8/BiasAddBiasAdd$cnn_block_7/conv2d_8/Conv2D:output:03cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_7/conv2d_8/BiasAdd?
0cnn_block_7/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_7_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_7/batch_normalization_7/ReadVariableOp?
2cnn_block_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_7/batch_normalization_7/ReadVariableOp_1?
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_7/conv2d_8/BiasAdd:output:08cnn_block_7/batch_normalization_7/ReadVariableOp:value:0:cnn_block_7/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 24
2cnn_block_7/batch_normalization_7/FusedBatchNormV3?
cnn_block_7/ReluRelu6cnn_block_7/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_7/Relu?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinput_tensor'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
conv2d_10/BiasAdd?
addAddV2cnn_block_7/Relu:activations:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????OO?2
add?
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_8_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp?
cnn_block_8/conv2d_9/Conv2DConv2Dadd:z:02cnn_block_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?*
paddingSAME*
strides
2
cnn_block_8/conv2d_9/Conv2D?
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp?
cnn_block_8/conv2d_9/BiasAddBiasAdd$cnn_block_8/conv2d_9/Conv2D:output:03cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_8/conv2d_9/BiasAdd?
0cnn_block_8/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_8_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype022
0cnn_block_8/batch_normalization_8/ReadVariableOp?
2cnn_block_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2cnn_block_8/batch_normalization_8/ReadVariableOp_1?
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_8/conv2d_9/BiasAdd:output:08cnn_block_8/batch_normalization_8/ReadVariableOp:value:0:cnn_block_8/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????OO?:?:?:?:?:*
epsilon%o?:*
is_training( 24
2cnn_block_8/batch_normalization_8/FusedBatchNormV3?
cnn_block_8/ReluRelu6cnn_block_8/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????OO?2
cnn_block_8/Relu?
max_pooling2d_1/MaxPoolMaxPoolcnn_block_8/Relu:activations:0*0
_output_shapes
:?????????''?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?	
IdentityIdentity max_pooling2d_1/MaxPool:output:0B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^cnn_block_6/batch_normalization_6/ReadVariableOp3^cnn_block_6/batch_normalization_6/ReadVariableOp_1,^cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+^cnn_block_6/conv2d_7/Conv2D/ReadVariableOpB^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^cnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^cnn_block_7/batch_normalization_7/ReadVariableOp3^cnn_block_7/batch_normalization_7/ReadVariableOp_1,^cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+^cnn_block_7/conv2d_8/Conv2D/ReadVariableOpB^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^cnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^cnn_block_8/batch_normalization_8/ReadVariableOp3^cnn_block_8/batch_normalization_8/ReadVariableOp_1,^cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+^cnn_block_8/conv2d_9/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????''?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????OO@::::::::::::::::::::2?
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_6/batch_normalization_6/ReadVariableOp0cnn_block_6/batch_normalization_6/ReadVariableOp2h
2cnn_block_6/batch_normalization_6/ReadVariableOp_12cnn_block_6/batch_normalization_6/ReadVariableOp_12Z
+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp+cnn_block_6/conv2d_7/BiasAdd/ReadVariableOp2X
*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp*cnn_block_6/conv2d_7/Conv2D/ReadVariableOp2?
Acnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAcnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_7/batch_normalization_7/ReadVariableOp0cnn_block_7/batch_normalization_7/ReadVariableOp2h
2cnn_block_7/batch_normalization_7/ReadVariableOp_12cnn_block_7/batch_normalization_7/ReadVariableOp_12Z
+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp+cnn_block_7/conv2d_8/BiasAdd/ReadVariableOp2X
*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp*cnn_block_7/conv2d_8/Conv2D/ReadVariableOp2?
Acnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpAcnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_8/batch_normalization_8/ReadVariableOp0cnn_block_8/batch_normalization_8/ReadVariableOp2h
2cnn_block_8/batch_normalization_8/ReadVariableOp_12cnn_block_8/batch_normalization_8/ReadVariableOp_12Z
+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp+cnn_block_8/conv2d_9/BiasAdd/ReadVariableOp2X
*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp*cnn_block_8/conv2d_9/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:?????????OO@
&
_user_specified_nameinput_tensor
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8291

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
H
,__inference_max_pooling2d_layer_call_fn_5698

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_56922
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_model_1_layer_call_fn_7100
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_70132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
4__inference_batch_normalization_7_layer_call_fn_8514

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_58642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
C__inference_res_block_layer_call_and_return_conditional_losses_7766
input_tensor7
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource8
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource=
9cnn_block_3_batch_normalization_3_readvariableop_resource?
;cnn_block_3_batch_normalization_3_readvariableop_1_resourceN
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource8
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource=
9cnn_block_4_batch_normalization_4_readvariableop_resource?
;cnn_block_4_batch_normalization_4_readvariableop_1_resourceN
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource7
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource8
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource=
9cnn_block_5_batch_normalization_5_readvariableop_resource?
;cnn_block_5_batch_normalization_5_readvariableop_1_resourceN
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity??0cnn_block_3/batch_normalization_3/AssignNewValue?2cnn_block_3/batch_normalization_3/AssignNewValue_1?Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_3/batch_normalization_3/ReadVariableOp?2cnn_block_3/batch_normalization_3/ReadVariableOp_1?+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?0cnn_block_4/batch_normalization_4/AssignNewValue?2cnn_block_4/batch_normalization_4/AssignNewValue_1?Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_4/batch_normalization_4/ReadVariableOp?2cnn_block_4/batch_normalization_4/ReadVariableOp_1?+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?0cnn_block_5/batch_normalization_5/AssignNewValue?2cnn_block_5/batch_normalization_5/AssignNewValue_1?Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_5/batch_normalization_5/ReadVariableOp?2cnn_block_5/batch_normalization_5/ReadVariableOp_1?+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?
cnn_block_3/conv2d_3/Conv2DConv2Dinput_tensor2cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
cnn_block_3/conv2d_3/Conv2D?
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_3/conv2d_3/BiasAdd?
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_3/batch_normalization_3/ReadVariableOp?
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_3/batch_normalization_3/ReadVariableOp_1?
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_3/batch_normalization_3/FusedBatchNormV3?
0cnn_block_3/batch_normalization_3/AssignNewValueAssignVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_3/batch_normalization_3/AssignNewValue?
2cnn_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_3/batch_normalization_3/AssignNewValue_1?
cnn_block_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_3/Relu?
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?
cnn_block_4/conv2d_4/Conv2DConv2Dcnn_block_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
cnn_block_4/conv2d_4/Conv2D?
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_4/conv2d_4/BiasAdd?
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_4/batch_normalization_4/ReadVariableOp?
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_4/batch_normalization_4/ReadVariableOp_1?
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_4/batch_normalization_4/FusedBatchNormV3?
0cnn_block_4/batch_normalization_4/AssignNewValueAssignVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource?cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_4/batch_normalization_4/AssignNewValue?
2cnn_block_4/batch_normalization_4/AssignNewValue_1AssignVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0D^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_4/batch_normalization_4/AssignNewValue_1?
cnn_block_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_4/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_6/BiasAdd?
addAddV2cnn_block_4/Relu:activations:0conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
add?
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?
cnn_block_5/conv2d_5/Conv2DConv2Dadd:z:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
cnn_block_5/conv2d_5/Conv2D?
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
cnn_block_5/conv2d_5/BiasAdd?
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_5/batch_normalization_5/ReadVariableOp?
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_5/batch_normalization_5/ReadVariableOp_1?
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_5/batch_normalization_5/FusedBatchNormV3?
0cnn_block_5/batch_normalization_5/AssignNewValueAssignVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_5/batch_normalization_5/AssignNewValue?
2cnn_block_5/batch_normalization_5/AssignNewValue_1AssignVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_5/batch_normalization_5/AssignNewValue_1?
cnn_block_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
cnn_block_5/Relu?
max_pooling2d/MaxPoolMaxPoolcnn_block_5/Relu:activations:0*/
_output_shapes
:?????????OO@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
IdentityIdentitymax_pooling2d/MaxPool:output:01^cnn_block_3/batch_normalization_3/AssignNewValue3^cnn_block_3/batch_normalization_3/AssignNewValue_1B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^cnn_block_3/batch_normalization_3/ReadVariableOp3^cnn_block_3/batch_normalization_3/ReadVariableOp_1,^cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+^cnn_block_3/conv2d_3/Conv2D/ReadVariableOp1^cnn_block_4/batch_normalization_4/AssignNewValue3^cnn_block_4/batch_normalization_4/AssignNewValue_1B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^cnn_block_4/batch_normalization_4/ReadVariableOp3^cnn_block_4/batch_normalization_4/ReadVariableOp_1,^cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+^cnn_block_4/conv2d_4/Conv2D/ReadVariableOp1^cnn_block_5/batch_normalization_5/AssignNewValue3^cnn_block_5/batch_normalization_5/AssignNewValue_1B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^cnn_block_5/batch_normalization_5/ReadVariableOp3^cnn_block_5/batch_normalization_5/ReadVariableOp_1,^cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+^cnn_block_5/conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????OO@2

Identity"
identityIdentity:output:0*?
_input_shapeso
m:???????????::::::::::::::::::::2d
0cnn_block_3/batch_normalization_3/AssignNewValue0cnn_block_3/batch_normalization_3/AssignNewValue2h
2cnn_block_3/batch_normalization_3/AssignNewValue_12cnn_block_3/batch_normalization_3/AssignNewValue_12?
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_3/batch_normalization_3/ReadVariableOp0cnn_block_3/batch_normalization_3/ReadVariableOp2h
2cnn_block_3/batch_normalization_3/ReadVariableOp_12cnn_block_3/batch_normalization_3/ReadVariableOp_12Z
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2X
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2d
0cnn_block_4/batch_normalization_4/AssignNewValue0cnn_block_4/batch_normalization_4/AssignNewValue2h
2cnn_block_4/batch_normalization_4/AssignNewValue_12cnn_block_4/batch_normalization_4/AssignNewValue_12?
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_4/batch_normalization_4/ReadVariableOp0cnn_block_4/batch_normalization_4/ReadVariableOp2h
2cnn_block_4/batch_normalization_4/ReadVariableOp_12cnn_block_4/batch_normalization_4/ReadVariableOp_12Z
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2X
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2d
0cnn_block_5/batch_normalization_5/AssignNewValue0cnn_block_5/batch_normalization_5/AssignNewValue2h
2cnn_block_5/batch_normalization_5/AssignNewValue_12cnn_block_5/batch_normalization_5/AssignNewValue_12?
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_5/batch_normalization_5/ReadVariableOp0cnn_block_5/batch_normalization_5/ReadVariableOp2h
2cnn_block_5/batch_normalization_5/ReadVariableOp_12cnn_block_5/batch_normalization_5/ReadVariableOp_12Z
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2X
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinput_tensor
?
D
(__inference_flatten_1_layer_call_fn_8188

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_66102
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????''?:X T
0
_output_shapes
:?????????''?
 
_user_specified_nameinputs
?
{
&__inference_dense_2_layer_call_fn_8207

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_66282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_5864

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5644

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
C__inference_res_block_layer_call_and_return_conditional_losses_6107
input_tensor7
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource8
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource=
9cnn_block_3_batch_normalization_3_readvariableop_resource?
;cnn_block_3_batch_normalization_3_readvariableop_1_resourceN
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource8
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource=
9cnn_block_4_batch_normalization_4_readvariableop_resource?
;cnn_block_4_batch_normalization_4_readvariableop_1_resourceN
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource7
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource8
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource=
9cnn_block_5_batch_normalization_5_readvariableop_resource?
;cnn_block_5_batch_normalization_5_readvariableop_1_resourceN
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity??0cnn_block_3/batch_normalization_3/AssignNewValue?2cnn_block_3/batch_normalization_3/AssignNewValue_1?Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_3/batch_normalization_3/ReadVariableOp?2cnn_block_3/batch_normalization_3/ReadVariableOp_1?+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?0cnn_block_4/batch_normalization_4/AssignNewValue?2cnn_block_4/batch_normalization_4/AssignNewValue_1?Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_4/batch_normalization_4/ReadVariableOp?2cnn_block_4/batch_normalization_4/ReadVariableOp_1?+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?0cnn_block_5/batch_normalization_5/AssignNewValue?2cnn_block_5/batch_normalization_5/AssignNewValue_1?Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0cnn_block_5/batch_normalization_5/ReadVariableOp?2cnn_block_5/batch_normalization_5/ReadVariableOp_1?+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp?
cnn_block_3/conv2d_3/Conv2DConv2Dinput_tensor2cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
cnn_block_3/conv2d_3/Conv2D?
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp?
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_3/conv2d_3/BiasAdd?
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_3/batch_normalization_3/ReadVariableOp?
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_3/batch_normalization_3/ReadVariableOp_1?
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_3/batch_normalization_3/FusedBatchNormV3?
0cnn_block_3/batch_normalization_3/AssignNewValueAssignVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_3/batch_normalization_3/AssignNewValue?
2cnn_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_3/batch_normalization_3/AssignNewValue_1?
cnn_block_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_3/Relu?
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp?
cnn_block_4/conv2d_4/Conv2DConv2Dcnn_block_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
cnn_block_4/conv2d_4/Conv2D?
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp?
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_4/conv2d_4/BiasAdd?
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_4/batch_normalization_4/ReadVariableOp?
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_4/batch_normalization_4/ReadVariableOp_1?
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_4/batch_normalization_4/FusedBatchNormV3?
0cnn_block_4/batch_normalization_4/AssignNewValueAssignVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource?cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_4/batch_normalization_4/AssignNewValue?
2cnn_block_4/batch_normalization_4/AssignNewValue_1AssignVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0D^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_4/batch_normalization_4/AssignNewValue_1?
cnn_block_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
cnn_block_4/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_6/BiasAdd?
addAddV2cnn_block_4/Relu:activations:0conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
add?
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp?
cnn_block_5/conv2d_5/Conv2DConv2Dadd:z:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
cnn_block_5/conv2d_5/Conv2D?
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp?
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
cnn_block_5/conv2d_5/BiasAdd?
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_5/batch_normalization_5/ReadVariableOp?
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_5/batch_normalization_5/ReadVariableOp_1?
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<24
2cnn_block_5/batch_normalization_5/FusedBatchNormV3?
0cnn_block_5/batch_normalization_5/AssignNewValueAssignVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*]
_classS
QOloc:@cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_5/batch_normalization_5/AssignNewValue?
2cnn_block_5/batch_normalization_5/AssignNewValue_1AssignVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*_
_classU
SQloc:@cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_5/batch_normalization_5/AssignNewValue_1?
cnn_block_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
cnn_block_5/Relu?
max_pooling2d/MaxPoolMaxPoolcnn_block_5/Relu:activations:0*/
_output_shapes
:?????????OO@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
IdentityIdentitymax_pooling2d/MaxPool:output:01^cnn_block_3/batch_normalization_3/AssignNewValue3^cnn_block_3/batch_normalization_3/AssignNewValue_1B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^cnn_block_3/batch_normalization_3/ReadVariableOp3^cnn_block_3/batch_normalization_3/ReadVariableOp_1,^cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+^cnn_block_3/conv2d_3/Conv2D/ReadVariableOp1^cnn_block_4/batch_normalization_4/AssignNewValue3^cnn_block_4/batch_normalization_4/AssignNewValue_1B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^cnn_block_4/batch_normalization_4/ReadVariableOp3^cnn_block_4/batch_normalization_4/ReadVariableOp_1,^cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+^cnn_block_4/conv2d_4/Conv2D/ReadVariableOp1^cnn_block_5/batch_normalization_5/AssignNewValue3^cnn_block_5/batch_normalization_5/AssignNewValue_1B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^cnn_block_5/batch_normalization_5/ReadVariableOp3^cnn_block_5/batch_normalization_5/ReadVariableOp_1,^cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+^cnn_block_5/conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????OO@2

Identity"
identityIdentity:output:0*?
_input_shapeso
m:???????????::::::::::::::::::::2d
0cnn_block_3/batch_normalization_3/AssignNewValue0cnn_block_3/batch_normalization_3/AssignNewValue2h
2cnn_block_3/batch_normalization_3/AssignNewValue_12cnn_block_3/batch_normalization_3/AssignNewValue_12?
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_3/batch_normalization_3/ReadVariableOp0cnn_block_3/batch_normalization_3/ReadVariableOp2h
2cnn_block_3/batch_normalization_3/ReadVariableOp_12cnn_block_3/batch_normalization_3/ReadVariableOp_12Z
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp2X
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp2d
0cnn_block_4/batch_normalization_4/AssignNewValue0cnn_block_4/batch_normalization_4/AssignNewValue2h
2cnn_block_4/batch_normalization_4/AssignNewValue_12cnn_block_4/batch_normalization_4/AssignNewValue_12?
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_4/batch_normalization_4/ReadVariableOp0cnn_block_4/batch_normalization_4/ReadVariableOp2h
2cnn_block_4/batch_normalization_4/ReadVariableOp_12cnn_block_4/batch_normalization_4/ReadVariableOp_12Z
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp2X
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp2d
0cnn_block_5/batch_normalization_5/AssignNewValue0cnn_block_5/batch_normalization_5/AssignNewValue2h
2cnn_block_5/batch_normalization_5/AssignNewValue_12cnn_block_5/batch_normalization_5/AssignNewValue_12?
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0cnn_block_5/batch_normalization_5/ReadVariableOp0cnn_block_5/batch_normalization_5/ReadVariableOp2h
2cnn_block_5/batch_normalization_5/ReadVariableOp_12cnn_block_5/batch_normalization_5/ReadVariableOp_12Z
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp2X
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinput_tensor
?
?
4__inference_batch_normalization_5_layer_call_fn_8399

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_56752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
(__inference_res_block_layer_call_fn_7886
input_tensor
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OO@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_res_block_layer_call_and_return_conditional_losses_61072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OO@2

Identity"
identityIdentity:output:0*?
_input_shapeso
m:???????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinput_tensor
?
?
4__inference_batch_normalization_8_layer_call_fn_8578

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_59682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_6_layer_call_fn_8463

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_5_layer_call_fn_8386

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_56442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5968

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_6610

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????''?:X T
0
_output_shapes
:?????????''?
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????;
dense_20
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_network?{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 158, 158, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ResBlock", "config": {"layer was saved without config": true}, "name": "res_block", "inbound_nodes": [[["input_1", 0, 0, {"training": false}]]]}, {"class_name": "ResBlock", "config": {"layer was saved without config": true}, "name": "res_block_1", "inbound_nodes": [[["res_block", 0, 0, {"training": false}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["res_block_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 158, 158, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 158, 158, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 158, 158, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 158, 158, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
channels
cnn1
cnn2
cnn3
pooling
identity_mapping
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ResBlock", "name": "res_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
channels
cnn1
cnn2
cnn3
pooling
identity_mapping
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ResBlock", "name": "res_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
 trainable_variables
!	variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 389376}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 389376]}}
?
*iter

+beta_1

,beta_2
	-decay
.learning_rate$m?%m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?$v?%v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?"
	optimizer
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
=14
>15
?16
@17
A18
B19
C20
D21
E22
F23
G24
H25
I26
J27
$28
%29"
trackable_list_wrapper
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
K14
L15
M16
N17
O18
P19
=20
>21
?22
@23
A24
B25
C26
D27
E28
F29
G30
H31
I32
J33
Q34
R35
S36
T37
U38
V39
$40
%41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
Wmetrics
Xnon_trainable_variables
	variables
Ylayer_metrics

Zlayers
	regularization_losses
[layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
?
\conv
]bn
^trainable_variables
_	variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "CNNBlock", "name": "cnn_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
bconv
cbn
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "CNNBlock", "name": "cnn_block_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
hconv
ibn
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "CNNBlock", "name": "cnn_block_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

;kernel
<bias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 158, 158, 1]}}
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13"
trackable_list_wrapper
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
K14
L15
M16
N17
O18
P19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
vmetrics
wnon_trainable_variables
xlayer_metrics
	variables

ylayers
regularization_losses
zlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
{conv
|bn
}trainable_variables
~	variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "CNNBlock", "name": "cnn_block_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
	?conv
?bn
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "CNNBlock", "name": "cnn_block_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
	?conv
?bn
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "CNNBlock", "name": "cnn_block_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Ikernel
Jbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 79, 79, 64]}}
?
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13"
trackable_list_wrapper
?
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
Q14
R15
S16
T17
U18
V19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
	variables
?layers
regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
!	variables
?layers
"regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??
2dense_2/kernel
:
2dense_2/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
'	variables
?layers
(regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?:= 2%res_block/cnn_block_3/conv2d_3/kernel
1:/ 2#res_block/cnn_block_3/conv2d_3/bias
?:= 21res_block/cnn_block_3/batch_normalization_3/gamma
>:< 20res_block/cnn_block_3/batch_normalization_3/beta
?:=  2%res_block/cnn_block_4/conv2d_4/kernel
1:/ 2#res_block/cnn_block_4/conv2d_4/bias
?:= 21res_block/cnn_block_4/batch_normalization_4/gamma
>:< 20res_block/cnn_block_4/batch_normalization_4/beta
?:= @2%res_block/cnn_block_5/conv2d_5/kernel
1:/@2#res_block/cnn_block_5/conv2d_5/bias
?:=@21res_block/cnn_block_5/batch_normalization_5/gamma
>:<@20res_block/cnn_block_5/batch_normalization_5/beta
3:1 2res_block/conv2d_6/kernel
%:# 2res_block/conv2d_6/bias
B:@@?2'res_block_1/cnn_block_6/conv2d_7/kernel
4:2?2%res_block_1/cnn_block_6/conv2d_7/bias
B:@?23res_block_1/cnn_block_6/batch_normalization_6/gamma
A:??22res_block_1/cnn_block_6/batch_normalization_6/beta
C:A??2'res_block_1/cnn_block_7/conv2d_8/kernel
4:2?2%res_block_1/cnn_block_7/conv2d_8/bias
B:@?23res_block_1/cnn_block_7/batch_normalization_7/gamma
A:??22res_block_1/cnn_block_7/batch_normalization_7/beta
C:A??2'res_block_1/cnn_block_8/conv2d_9/kernel
4:2?2%res_block_1/cnn_block_8/conv2d_9/bias
B:@?23res_block_1/cnn_block_8/batch_normalization_8/gamma
A:??22res_block_1/cnn_block_8/batch_normalization_8/beta
7:5@?2res_block_1/conv2d_10/kernel
):'?2res_block_1/conv2d_10/bias
G:E  (27res_block/cnn_block_3/batch_normalization_3/moving_mean
K:I  (2;res_block/cnn_block_3/batch_normalization_3/moving_variance
G:E  (27res_block/cnn_block_4/batch_normalization_4/moving_mean
K:I  (2;res_block/cnn_block_4/batch_normalization_4/moving_variance
G:E@ (27res_block/cnn_block_5/batch_normalization_5/moving_mean
K:I@ (2;res_block/cnn_block_5/batch_normalization_5/moving_variance
J:H? (29res_block_1/cnn_block_6/batch_normalization_6/moving_mean
N:L? (2=res_block_1/cnn_block_6/batch_normalization_6/moving_variance
J:H? (29res_block_1/cnn_block_7/batch_normalization_7/moving_mean
N:L? (2=res_block_1/cnn_block_7/batch_normalization_7/moving_variance
J:H? (29res_block_1/cnn_block_8/batch_normalization_8/moving_mean
N:L? (2=res_block_1/cnn_block_8/batch_normalization_8/moving_variance
0
?0
?1"
trackable_list_wrapper
v
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
?	

/kernel
0bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 158, 158, 1]}}
?	
	?axis
	1gamma
2beta
Kmoving_mean
Lmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 158, 158, 32]}}
<
/0
01
12
23"
trackable_list_wrapper
J
/0
01
12
23
K4
L5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
_	variables
?layers
`regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

3kernel
4bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 158, 158, 32]}}
?	
	?axis
	5gamma
6beta
Mmoving_mean
Nmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 158, 158, 32]}}
<
30
41
52
63"
trackable_list_wrapper
J
30
41
52
63
M4
N5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
dtrainable_variables
?metrics
?non_trainable_variables
?layer_metrics
e	variables
?layers
fregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

7kernel
8bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 158, 158, 32]}}
?	
	?axis
	9gamma
:beta
Omoving_mean
Pmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 158, 158, 64]}}
<
70
81
92
:3"
trackable_list_wrapper
J
70
81
92
:3
O4
P5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jtrainable_variables
?metrics
?non_trainable_variables
?layer_metrics
k	variables
?layers
lregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ntrainable_variables
?metrics
?non_trainable_variables
?layer_metrics
o	variables
?layers
pregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rtrainable_variables
?metrics
?non_trainable_variables
?layer_metrics
s	variables
?layers
tregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
K0
L1
M2
N3
O4
P5"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
?	

=kernel
>bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 79, 79, 64]}}
?	
	?axis
	?gamma
@beta
Qmoving_mean
Rmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 79, 79, 128]}}
<
=0
>1
?2
@3"
trackable_list_wrapper
J
=0
>1
?2
@3
Q4
R5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
~	variables
?layers
regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

Akernel
Bbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 79, 79, 128]}}
?	
	?axis
	Cgamma
Dbeta
Smoving_mean
Tmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 79, 79, 128]}}
<
A0
B1
C2
D3"
trackable_list_wrapper
J
A0
B1
C2
D3
S4
T5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

Ekernel
Fbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 79, 79, 128]}}
?	
	?axis
	Ggamma
Hbeta
Umoving_mean
Vmoving_variance
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 79, 79, 256]}}
<
E0
F1
G2
H3"
trackable_list_wrapper
J
E0
F1
G2
H3
U4
V5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
Q0
R1
S2
T3
U4
V5"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
<
10
21
K2
L3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
<
50
61
M2
N3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
<
90
:1
O2
P3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
<
?0
@1
Q2
R3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
<
C0
D1
S2
T3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
<
G0
H1
U2
V3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?layer_metrics
?	variables
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
':%
??
2Adam/dense_2/kernel/m
:
2Adam/dense_2/bias/m
D:B 2,Adam/res_block/cnn_block_3/conv2d_3/kernel/m
6:4 2*Adam/res_block/cnn_block_3/conv2d_3/bias/m
D:B 28Adam/res_block/cnn_block_3/batch_normalization_3/gamma/m
C:A 27Adam/res_block/cnn_block_3/batch_normalization_3/beta/m
D:B  2,Adam/res_block/cnn_block_4/conv2d_4/kernel/m
6:4 2*Adam/res_block/cnn_block_4/conv2d_4/bias/m
D:B 28Adam/res_block/cnn_block_4/batch_normalization_4/gamma/m
C:A 27Adam/res_block/cnn_block_4/batch_normalization_4/beta/m
D:B @2,Adam/res_block/cnn_block_5/conv2d_5/kernel/m
6:4@2*Adam/res_block/cnn_block_5/conv2d_5/bias/m
D:B@28Adam/res_block/cnn_block_5/batch_normalization_5/gamma/m
C:A@27Adam/res_block/cnn_block_5/batch_normalization_5/beta/m
8:6 2 Adam/res_block/conv2d_6/kernel/m
*:( 2Adam/res_block/conv2d_6/bias/m
G:E@?2.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/m
9:7?2,Adam/res_block_1/cnn_block_6/conv2d_7/bias/m
G:E?2:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/m
F:D?29Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/m
H:F??2.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/m
9:7?2,Adam/res_block_1/cnn_block_7/conv2d_8/bias/m
G:E?2:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/m
F:D?29Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/m
H:F??2.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/m
9:7?2,Adam/res_block_1/cnn_block_8/conv2d_9/bias/m
G:E?2:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/m
F:D?29Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/m
<::@?2#Adam/res_block_1/conv2d_10/kernel/m
.:,?2!Adam/res_block_1/conv2d_10/bias/m
':%
??
2Adam/dense_2/kernel/v
:
2Adam/dense_2/bias/v
D:B 2,Adam/res_block/cnn_block_3/conv2d_3/kernel/v
6:4 2*Adam/res_block/cnn_block_3/conv2d_3/bias/v
D:B 28Adam/res_block/cnn_block_3/batch_normalization_3/gamma/v
C:A 27Adam/res_block/cnn_block_3/batch_normalization_3/beta/v
D:B  2,Adam/res_block/cnn_block_4/conv2d_4/kernel/v
6:4 2*Adam/res_block/cnn_block_4/conv2d_4/bias/v
D:B 28Adam/res_block/cnn_block_4/batch_normalization_4/gamma/v
C:A 27Adam/res_block/cnn_block_4/batch_normalization_4/beta/v
D:B @2,Adam/res_block/cnn_block_5/conv2d_5/kernel/v
6:4@2*Adam/res_block/cnn_block_5/conv2d_5/bias/v
D:B@28Adam/res_block/cnn_block_5/batch_normalization_5/gamma/v
C:A@27Adam/res_block/cnn_block_5/batch_normalization_5/beta/v
8:6 2 Adam/res_block/conv2d_6/kernel/v
*:( 2Adam/res_block/conv2d_6/bias/v
G:E@?2.Adam/res_block_1/cnn_block_6/conv2d_7/kernel/v
9:7?2,Adam/res_block_1/cnn_block_6/conv2d_7/bias/v
G:E?2:Adam/res_block_1/cnn_block_6/batch_normalization_6/gamma/v
F:D?29Adam/res_block_1/cnn_block_6/batch_normalization_6/beta/v
H:F??2.Adam/res_block_1/cnn_block_7/conv2d_8/kernel/v
9:7?2,Adam/res_block_1/cnn_block_7/conv2d_8/bias/v
G:E?2:Adam/res_block_1/cnn_block_7/batch_normalization_7/gamma/v
F:D?29Adam/res_block_1/cnn_block_7/batch_normalization_7/beta/v
H:F??2.Adam/res_block_1/cnn_block_8/conv2d_9/kernel/v
9:7?2,Adam/res_block_1/cnn_block_8/conv2d_9/bias/v
G:E?2:Adam/res_block_1/cnn_block_8/batch_normalization_8/gamma/v
F:D?29Adam/res_block_1/cnn_block_8/batch_normalization_8/beta/v
<::@?2#Adam/res_block_1/conv2d_10/kernel/v
.:,?2!Adam/res_block_1/conv2d_10/bias/v
?2?
__inference__wrapped_model_5374?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_1???????????
?2?
&__inference_model_1_layer_call_fn_7596
&__inference_model_1_layer_call_fn_7100
&__inference_model_1_layer_call_fn_7685
&__inference_model_1_layer_call_fn_6919?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_1_layer_call_and_return_conditional_losses_7507
A__inference_model_1_layer_call_and_return_conditional_losses_6645
A__inference_model_1_layer_call_and_return_conditional_losses_6737
A__inference_model_1_layer_call_and_return_conditional_losses_7353?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_res_block_layer_call_fn_7886
(__inference_res_block_layer_call_fn_7931?
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_res_block_layer_call_and_return_conditional_losses_7841
C__inference_res_block_layer_call_and_return_conditional_losses_7766?
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_res_block_1_layer_call_fn_8132
*__inference_res_block_1_layer_call_fn_8177?
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_res_block_1_layer_call_and_return_conditional_losses_8087
E__inference_res_block_1_layer_call_and_return_conditional_losses_8012?
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_flatten_1_layer_call_fn_8188?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_1_layer_call_and_return_conditional_losses_8183?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_8207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_8198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_7199input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_max_pooling2d_layer_call_fn_5698?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5692?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_max_pooling2d_1_layer_call_fn_6022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6016?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_3_layer_call_fn_8258
4__inference_batch_normalization_3_layer_call_fn_8271?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8227
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8245?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_4_layer_call_fn_8322
4__inference_batch_normalization_4_layer_call_fn_8335?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8291
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8309?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_5_layer_call_fn_8399
4__inference_batch_normalization_5_layer_call_fn_8386?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8355
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8373?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_6_layer_call_fn_8450
4__inference_batch_normalization_6_layer_call_fn_8463?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_8419
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_8437?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_7_layer_call_fn_8514
4__inference_batch_normalization_7_layer_call_fn_8527?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8501
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8483?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_8_layer_call_fn_8578
4__inference_batch_normalization_8_layer_call_fn_8591?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_8565
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_8547?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
__inference__wrapped_model_5374?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%:?7
0?-
+?(
input_1???????????
? "1?.
,
dense_2!?
dense_2?????????
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8227?12KLM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8245?12KLM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_batch_normalization_3_layer_call_fn_8258?12KLM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_3_layer_call_fn_8271?12KLM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8291?56MNM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8309?56MNM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_batch_normalization_4_layer_call_fn_8322?56MNM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
4__inference_batch_normalization_4_layer_call_fn_8335?56MNM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8355?9:OPM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8373?9:OPM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
4__inference_batch_normalization_5_layer_call_fn_8386?9:OPM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
4__inference_batch_normalization_5_layer_call_fn_8399?9:OPM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_8419??@QRN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_8437??@QRN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
4__inference_batch_normalization_6_layer_call_fn_8450??@QRN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
4__inference_batch_normalization_6_layer_call_fn_8463??@QRN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8483?CDSTN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8501?CDSTN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
4__inference_batch_normalization_7_layer_call_fn_8514?CDSTN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
4__inference_batch_normalization_7_layer_call_fn_8527?CDSTN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_8547?GHUVN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_8565?GHUVN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
4__inference_batch_normalization_8_layer_call_fn_8578?GHUVN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
4__inference_batch_normalization_8_layer_call_fn_8591?GHUVN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
A__inference_dense_2_layer_call_and_return_conditional_losses_8198^$%1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????

? {
&__inference_dense_2_layer_call_fn_8207Q$%1?.
'?$
"?
inputs???????????
? "??????????
?
C__inference_flatten_1_layer_call_and_return_conditional_losses_8183c8?5
.?+
)?&
inputs?????????''?
? "'?$
?
0???????????
? ?
(__inference_flatten_1_layer_call_fn_8188V8?5
.?+
)?&
inputs?????????''?
? "?????????????
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6016?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_1_layer_call_fn_6022?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5692?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_max_pooling2d_layer_call_fn_5698?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_model_1_layer_call_and_return_conditional_losses_6645?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%B??
8?5
+?(
input_1???????????
p

 
? "%?"
?
0?????????

? ?
A__inference_model_1_layer_call_and_return_conditional_losses_6737?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%B??
8?5
+?(
input_1???????????
p 

 
? "%?"
?
0?????????

? ?
A__inference_model_1_layer_call_and_return_conditional_losses_7353?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????

? ?
A__inference_model_1_layer_call_and_return_conditional_losses_7507?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????

? ?
&__inference_model_1_layer_call_fn_6919?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%B??
8?5
+?(
input_1???????????
p

 
? "??????????
?
&__inference_model_1_layer_call_fn_7100?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%B??
8?5
+?(
input_1???????????
p 

 
? "??????????
?
&__inference_model_1_layer_call_fn_7596?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%A?>
7?4
*?'
inputs???????????
p

 
? "??????????
?
&__inference_model_1_layer_call_fn_7685?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%A?>
7?4
*?'
inputs???????????
p 

 
? "??????????
?
E__inference_res_block_1_layer_call_and_return_conditional_losses_8012?=>?@QRABCDSTIJEFGHUVA?>
7?4
.?+
input_tensor?????????OO@
p
? ".?+
$?!
0?????????''?
? ?
E__inference_res_block_1_layer_call_and_return_conditional_losses_8087?=>?@QRABCDSTIJEFGHUVA?>
7?4
.?+
input_tensor?????????OO@
p 
? ".?+
$?!
0?????????''?
? ?
*__inference_res_block_1_layer_call_fn_8132|=>?@QRABCDSTIJEFGHUVA?>
7?4
.?+
input_tensor?????????OO@
p
? "!??????????''??
*__inference_res_block_1_layer_call_fn_8177|=>?@QRABCDSTIJEFGHUVA?>
7?4
.?+
input_tensor?????????OO@
p 
? "!??????????''??
C__inference_res_block_layer_call_and_return_conditional_losses_7766?/012KL3456MN;<789:OPC?@
9?6
0?-
input_tensor???????????
p
? "-?*
#? 
0?????????OO@
? ?
C__inference_res_block_layer_call_and_return_conditional_losses_7841?/012KL3456MN;<789:OPC?@
9?6
0?-
input_tensor???????????
p 
? "-?*
#? 
0?????????OO@
? ?
(__inference_res_block_layer_call_fn_7886}/012KL3456MN;<789:OPC?@
9?6
0?-
input_tensor???????????
p
? " ??????????OO@?
(__inference_res_block_layer_call_fn_7931}/012KL3456MN;<789:OPC?@
9?6
0?-
input_tensor???????????
p 
? " ??????????OO@?
"__inference_signature_wrapper_7199?*/012KL3456MN;<789:OP=>?@QRABCDSTIJEFGHUV$%E?B
? 
;?8
6
input_1+?(
input_1???????????"1?.
,
dense_2!?
dense_2?????????
