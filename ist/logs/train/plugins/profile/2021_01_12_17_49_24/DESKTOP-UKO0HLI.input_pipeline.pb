  *	?????؈@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?lV}???!?v,??%N@)?ׁsF???1	B3??>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate?^)????!?x???O:@)㥛? ???1Hj5+9:@:Preprocessing2U
Iterator::Model::ParallelMapV2????K7??!??)C??0@)????K7??1??)C??0@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??n????!????l2@)??#?????1??-?n,$@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat~??k	???!??Y?Y? @)q???h??1?R??? @:Preprocessing2F
Iterator::Model?W?2??!??+??3@)$????ۗ?1??>?q@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate?~j?t???!???)?%@)
ףp=
??1pW?@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?<,Ԛ?}?!D@?Da??)?<,Ԛ?}?1D@?Da??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!?!??g\???)a2U0*?s?1p???}Q??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip~8gDi??!T??1?N@)ŏ1w-!o?1??g\???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!.;Z????)Ǻ???f?1.;Z????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range/n??b?!ST?]???)/n??b?1ST?]???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensor/n??R?!ST?]???)/n??R?1ST?]???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensorǺ???F?!.;Z????)Ǻ???F?1.;Z????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice-C??6:?!??4?????)-C??6:?1??4?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.