model = './test.prototxt';
caffe.set_mode_cpu();

% caffe.set_mode_gpu();
% caffe.set_device(gpu_id);
net = caffe.Net(model, 'test');

data = randn(net.blobs('data').shape);


net.blobs('data').set_data(data);
net.forward_prefilled();
res = net.blobs('testlayer').get_data();

diff = randn(net.blobs('testlayer').shape);
net.blobs('testlayer').set_diff(diff);
net.backward_prefilled();
data_diff = net.blobs('data').get_diff();

DistanceMetric = caffe.io.read_mean('DistanceMetric.blob');
TransferMatrix = caffe.io.read_mean('TransferMatrix.blob');
NormTransferMatrix = caffe.io.read_mean('NormTransferMatrix.blob');