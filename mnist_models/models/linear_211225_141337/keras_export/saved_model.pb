??
??
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
executor_typestring ??
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
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
}
dense_200/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*!
shared_namedense_200/kernel
v
$dense_200/kernel/Read/ReadVariableOpReadVariableOpdense_200/kernel*
_output_shapes
:	?
*
dtype0
t
dense_200/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_200/bias
m
"dense_200/bias/Read/ReadVariableOpReadVariableOpdense_200/bias*
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
Adam/dense_200/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*(
shared_nameAdam/dense_200/kernel/m
?
+Adam/dense_200/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_200/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/dense_200/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_200/bias/m
{
)Adam/dense_200/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_200/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_200/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*(
shared_nameAdam/dense_200/kernel/v
?
+Adam/dense_200/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_200/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/dense_200/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_200/bias/v
{
)Adam/dense_200/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_200/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
R

	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
d
iter

beta_1

beta_2
	decay
learning_ratem<m=v>v?

0
1

0
1
 
?
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
?
"non_trainable_variables

#layers
$metrics
%layer_regularization_losses
&layer_metrics

	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_200/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_200/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
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
 

0
1
2

10
21
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
 
 
 
 
 
 
4
	3total
	4count
5	variables
6	keras_api
D
	7total
	8count
9
_fn_kwargs
:	variables
;	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

30
41

5	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

:	variables
}
VARIABLE_VALUEAdam/dense_200/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_200/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_200/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_200/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
 serving_default_flatten_80_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_flatten_80_inputdense_200/kerneldense_200/bias*
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_184671
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_200/kernel/Read/ReadVariableOp"dense_200/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_200/kernel/m/Read/ReadVariableOp)Adam/dense_200/bias/m/Read/ReadVariableOp+Adam/dense_200/kernel/v/Read/ReadVariableOp)Adam/dense_200/bias/v/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_184823
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_200/kerneldense_200/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_200/kernel/mAdam/dense_200/bias/mAdam/dense_200/kernel/vAdam/dense_200/bias/v*
Tin
2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_184878??
?
?
.__inference_sequential_19_layer_call_fn_184680

inputs
unknown:	?

	unknown_0:

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
GPU 2J 8? *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_184565o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_80_layer_call_and_return_conditional_losses_184539

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184565

inputs#
dense_200_184552:	?

dense_200_184554:

identity??!dense_200/StatefulPartitionedCall?
flatten_80/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_184539?
!dense_200/StatefulPartitionedCallStatefulPartitionedCall#flatten_80/PartitionedCall:output:0dense_200_184552dense_200_184554*
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
GPU 2J 8? *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_184551?
softmax_80/PartitionedCallPartitionedCall*dense_200/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_80_layer_call_and_return_conditional_losses_184562r
IdentityIdentity#softmax_80/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
j
NoOpNoOp"^dense_200/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184715

inputs;
(dense_200_matmul_readvariableop_resource:	?
7
)dense_200_biasadd_readvariableop_resource:

identity?? dense_200/BiasAdd/ReadVariableOp?dense_200/MatMul/ReadVariableOpa
flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  s
flatten_80/ReshapeReshapeinputsflatten_80/Const:output:0*
T0*(
_output_shapes
:???????????
dense_200/MatMul/ReadVariableOpReadVariableOp(dense_200_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense_200/MatMulMatMulflatten_80/Reshape:output:0'dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_200/BiasAdd/ReadVariableOpReadVariableOp)dense_200_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_200/BiasAddBiasAdddense_200/MatMul:product:0(dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
k
softmax_80/SoftmaxSoftmaxdense_200/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
k
IdentityIdentitysoftmax_80/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp!^dense_200/BiasAdd/ReadVariableOp ^dense_200/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 dense_200/BiasAdd/ReadVariableOp dense_200/BiasAdd/ReadVariableOp2B
dense_200/MatMul/ReadVariableOpdense_200/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184616

inputs#
dense_200_184609:	?

dense_200_184611:

identity??!dense_200/StatefulPartitionedCall?
flatten_80/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_184539?
!dense_200/StatefulPartitionedCallStatefulPartitionedCall#flatten_80/PartitionedCall:output:0dense_200_184609dense_200_184611*
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
GPU 2J 8? *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_184551?
softmax_80/PartitionedCallPartitionedCall*dense_200/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_80_layer_call_and_return_conditional_losses_184562r
IdentityIdentity#softmax_80/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
j
NoOpNoOp"^dense_200/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_flatten_80_layer_call_fn_184720

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_184539a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_184526
flatten_80_inputI
6sequential_19_dense_200_matmul_readvariableop_resource:	?
E
7sequential_19_dense_200_biasadd_readvariableop_resource:

identity??.sequential_19/dense_200/BiasAdd/ReadVariableOp?-sequential_19/dense_200/MatMul/ReadVariableOpo
sequential_19/flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ?
 sequential_19/flatten_80/ReshapeReshapeflatten_80_input'sequential_19/flatten_80/Const:output:0*
T0*(
_output_shapes
:???????????
-sequential_19/dense_200/MatMul/ReadVariableOpReadVariableOp6sequential_19_dense_200_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
sequential_19/dense_200/MatMulMatMul)sequential_19/flatten_80/Reshape:output:05sequential_19/dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
.sequential_19/dense_200/BiasAdd/ReadVariableOpReadVariableOp7sequential_19_dense_200_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential_19/dense_200/BiasAddBiasAdd(sequential_19/dense_200/MatMul:product:06sequential_19/dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 sequential_19/softmax_80/SoftmaxSoftmax(sequential_19/dense_200/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
y
IdentityIdentity*sequential_19/softmax_80/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp/^sequential_19/dense_200/BiasAdd/ReadVariableOp.^sequential_19/dense_200/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2`
.sequential_19/dense_200/BiasAdd/ReadVariableOp.sequential_19/dense_200/BiasAdd/ReadVariableOp2^
-sequential_19/dense_200/MatMul/ReadVariableOp-sequential_19/dense_200/MatMul/ReadVariableOp:a ]
/
_output_shapes
:?????????
*
_user_specified_nameflatten_80_input
?'
?
__inference__traced_save_184823
file_prefix/
+savev2_dense_200_kernel_read_readvariableop-
)savev2_dense_200_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_200_kernel_m_read_readvariableop4
0savev2_adam_dense_200_bias_m_read_readvariableop6
2savev2_adam_dense_200_kernel_v_read_readvariableop4
0savev2_adam_dense_200_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_200_kernel_read_readvariableop)savev2_dense_200_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_200_kernel_m_read_readvariableop0savev2_adam_dense_200_bias_m_read_readvariableop2savev2_adam_dense_200_kernel_v_read_readvariableop0savev2_adam_dense_200_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*\
_input_shapesK
I: :	?
:
: : : : : : : : : :	?
:
:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?
: 

_output_shapes
:
:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: 
?	
?
E__inference_dense_200_layer_call_and_return_conditional_losses_184745

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?=
?
"__inference__traced_restore_184878
file_prefix4
!assignvariableop_dense_200_kernel:	?
/
!assignvariableop_1_dense_200_bias:
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: $
assignvariableop_9_total_1: %
assignvariableop_10_count_1: >
+assignvariableop_11_adam_dense_200_kernel_m:	?
7
)assignvariableop_12_adam_dense_200_bias_m:
>
+assignvariableop_13_adam_dense_200_kernel_v:	?
7
)assignvariableop_14_adam_dense_200_bias_v:

identity_16??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_dense_200_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_200_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_total_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_adam_dense_200_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_dense_200_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_dense_200_kernel_vIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_200_bias_vIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_16IdentityIdentity_15:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_16Identity_16:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
$__inference_signature_wrapper_184671
flatten_80_input
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_80_inputunknown	unknown_0*
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_184526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameflatten_80_input
?
b
F__inference_softmax_80_layer_call_and_return_conditional_losses_184562

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184643
flatten_80_input#
dense_200_184636:	?

dense_200_184638:

identity??!dense_200/StatefulPartitionedCall?
flatten_80/PartitionedCallPartitionedCallflatten_80_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_184539?
!dense_200/StatefulPartitionedCallStatefulPartitionedCall#flatten_80/PartitionedCall:output:0dense_200_184636dense_200_184638*
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
GPU 2J 8? *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_184551?
softmax_80/PartitionedCallPartitionedCall*dense_200/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_80_layer_call_and_return_conditional_losses_184562r
IdentityIdentity#softmax_80/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
j
NoOpNoOp"^dense_200/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameflatten_80_input
?	
?
E__inference_dense_200_layer_call_and_return_conditional_losses_184551

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_80_layer_call_and_return_conditional_losses_184726

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_200_layer_call_fn_184735

inputs
unknown:	?

	unknown_0:

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
GPU 2J 8? *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_184551o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_19_layer_call_fn_184572
flatten_80_input
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_80_inputunknown	unknown_0*
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
GPU 2J 8? *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_184565o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameflatten_80_input
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184702

inputs;
(dense_200_matmul_readvariableop_resource:	?
7
)dense_200_biasadd_readvariableop_resource:

identity?? dense_200/BiasAdd/ReadVariableOp?dense_200/MatMul/ReadVariableOpa
flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  s
flatten_80/ReshapeReshapeinputsflatten_80/Const:output:0*
T0*(
_output_shapes
:???????????
dense_200/MatMul/ReadVariableOpReadVariableOp(dense_200_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense_200/MatMulMatMulflatten_80/Reshape:output:0'dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_200/BiasAdd/ReadVariableOpReadVariableOp)dense_200_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_200/BiasAddBiasAdddense_200/MatMul:product:0(dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
k
softmax_80/SoftmaxSoftmaxdense_200/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
k
IdentityIdentitysoftmax_80/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp!^dense_200/BiasAdd/ReadVariableOp ^dense_200/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 dense_200/BiasAdd/ReadVariableOp dense_200/BiasAdd/ReadVariableOp2B
dense_200/MatMul/ReadVariableOpdense_200/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_softmax_80_layer_call_fn_184750

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_80_layer_call_and_return_conditional_losses_184562`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
.__inference_sequential_19_layer_call_fn_184689

inputs
unknown:	?

	unknown_0:

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
GPU 2J 8? *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_184616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_softmax_80_layer_call_and_return_conditional_losses_184755

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184654
flatten_80_input#
dense_200_184647:	?

dense_200_184649:

identity??!dense_200/StatefulPartitionedCall?
flatten_80/PartitionedCallPartitionedCallflatten_80_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_80_layer_call_and_return_conditional_losses_184539?
!dense_200/StatefulPartitionedCallStatefulPartitionedCall#flatten_80/PartitionedCall:output:0dense_200_184647dense_200_184649*
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
GPU 2J 8? *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_184551?
softmax_80/PartitionedCallPartitionedCall*dense_200/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_80_layer_call_and_return_conditional_losses_184562r
IdentityIdentity#softmax_80/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
j
NoOpNoOp"^dense_200/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameflatten_80_input
?
?
.__inference_sequential_19_layer_call_fn_184632
flatten_80_input
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_80_inputunknown	unknown_0*
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
GPU 2J 8? *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_184616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameflatten_80_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
flatten_80_inputA
"serving_default_flatten_80_input:0?????????>

softmax_800
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:?K
?
layer-0
layer_with_weights-0
layer-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
@__call__
*A&call_and_return_all_conditional_losses
B_default_save_signature"
_tf_keras_sequential
?

	variables
trainable_variables
regularization_losses
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
w
iter

beta_1

beta_2
	decay
learning_ratem<m=v>v?"
	optimizer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
@__call__
B_default_save_signature
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
,
Iserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"non_trainable_variables

#layers
$metrics
%layer_regularization_losses
&layer_metrics

	variables
trainable_variables
regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
#:!	?
2dense_200/kernel
:
2dense_200/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
10
21"
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
N
	3total
	4count
5	variables
6	keras_api"
_tf_keras_metric
^
	7total
	8count
9
_fn_kwargs
:	variables
;	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
30
41"
trackable_list_wrapper
-
5	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
-
:	variables"
_generic_user_object
(:&	?
2Adam/dense_200/kernel/m
!:
2Adam/dense_200/bias/m
(:&	?
2Adam/dense_200/kernel/v
!:
2Adam/dense_200/bias/v
?2?
.__inference_sequential_19_layer_call_fn_184572
.__inference_sequential_19_layer_call_fn_184680
.__inference_sequential_19_layer_call_fn_184689
.__inference_sequential_19_layer_call_fn_184632?
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
I__inference_sequential_19_layer_call_and_return_conditional_losses_184702
I__inference_sequential_19_layer_call_and_return_conditional_losses_184715
I__inference_sequential_19_layer_call_and_return_conditional_losses_184643
I__inference_sequential_19_layer_call_and_return_conditional_losses_184654?
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
?B?
!__inference__wrapped_model_184526flatten_80_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten_80_layer_call_fn_184720?
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
F__inference_flatten_80_layer_call_and_return_conditional_losses_184726?
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
*__inference_dense_200_layer_call_fn_184735?
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
E__inference_dense_200_layer_call_and_return_conditional_losses_184745?
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
+__inference_softmax_80_layer_call_fn_184750?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_softmax_80_layer_call_and_return_conditional_losses_184755?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_184671flatten_80_input"?
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
 ?
!__inference__wrapped_model_184526?A?>
7?4
2?/
flatten_80_input?????????
? "7?4
2

softmax_80$?!

softmax_80?????????
?
E__inference_dense_200_layer_call_and_return_conditional_losses_184745]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ~
*__inference_dense_200_layer_call_fn_184735P0?-
&?#
!?
inputs??????????
? "??????????
?
F__inference_flatten_80_layer_call_and_return_conditional_losses_184726a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
+__inference_flatten_80_layer_call_fn_184720T7?4
-?*
(?%
inputs?????????
? "????????????
I__inference_sequential_19_layer_call_and_return_conditional_losses_184643vI?F
??<
2?/
flatten_80_input?????????
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184654vI?F
??<
2?/
flatten_80_input?????????
p

 
? "%?"
?
0?????????

? ?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184702l??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_19_layer_call_and_return_conditional_losses_184715l??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
.__inference_sequential_19_layer_call_fn_184572iI?F
??<
2?/
flatten_80_input?????????
p 

 
? "??????????
?
.__inference_sequential_19_layer_call_fn_184632iI?F
??<
2?/
flatten_80_input?????????
p

 
? "??????????
?
.__inference_sequential_19_layer_call_fn_184680_??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
.__inference_sequential_19_layer_call_fn_184689_??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
$__inference_signature_wrapper_184671?U?R
? 
K?H
F
flatten_80_input2?/
flatten_80_input?????????"7?4
2

softmax_80$?!

softmax_80?????????
?
F__inference_softmax_80_layer_call_and_return_conditional_losses_184755\3?0
)?&
 ?
inputs?????????


 
? "%?"
?
0?????????

? ~
+__inference_softmax_80_layer_call_fn_184750O3?0
)?&
 ?
inputs?????????


 
? "??????????
