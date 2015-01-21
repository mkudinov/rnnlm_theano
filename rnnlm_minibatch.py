import numpy as np
import theano as T
import theano.tensor as TT

def get_minibatch_matrix(dataset, number, starting_point):
    batches = []
    new_batch = [0]
    current = 0
    if dataset[starting_point] == 0:
        starting_point += 1
    end_point = starting_point
    for index in dataset[starting_point:]:
        new_batch.append(index)
        end_point += 1
        if index == 0:
            batches.append(new_batch)
            new_batch = [0]
            current += 1
            if len(batches) == number:
                break
    max_len = max(len(x) for x in batches)
    batch_masks = []
    for l in batches:
        new_batch_mask = ([1] * (len(l) - 1)) + ([0] * (max_len - len(l)))
        batch_masks.append(new_batch_mask)
        if len(l) < max_len:
            l += ([0] * (max_len - len(l)))
    return np.transpose(np.asarray(batches, np.int32)), np.asarray(batch_masks, np.int32), end_point, len(batches)

np.set_printoptions(threshold=np.nan)

path = "Data/mikolov_corp.npz"
path_dic = "Data/mikolov_dict.npz"

#dictionary = np.load(path_dic)

print "Data loading"
data = np.load(path)
oov_code = data["oov"]
n_words = data["n_words"]

#train_data = np.append([0], data['train_words']).astype('int32')
#train_data = np.asarray([x for x in train_data if x != oov_code]).astype('int32')
#train_len = train_data.shape[0]

#valid_data = np.append([0], data["valid_words"]).astype('int32')
#valid_data = np.asarray([x for x in valid_data if x != oov_code]).astype('int32')
#valid_len = valid_data.shape[0]

train_data = data['train_words'].astype(np.int32)
train_len = train_data.shape[0]
valid_data = data['valid_words'].astype(np.int32)
valid_len = valid_data.shape[0]

n_in = n_words
n_hid = 100 
n_minibatches_num = 100

X = TT.matrix('X', 'int32')
Y = TT.matrix('Y', 'int32')
X_MASK = TT.matrix('X_MASK', 'int32')
lr = TT.scalar('lr', T.config.floatX)
mom = TT.scalar('lr', T.config.floatX)
n_minibatches = TT.scalar('n_minibatches', 'int32')
H0 = TT.matrix('H0', T.config.floatX)
print "Compilation"

W_init = np.random.uniform(-0.1, 0.1, (n_hid, n_in + 1)).astype(T.config.floatX)
W_init[:,oov_code] = np.asarray([0] * n_hid)
W_in = T.shared(W_init, borrow=True)
W_rec = T.shared(np.random.uniform(-0.1, 0.1, (n_hid, n_hid)).astype(T.config.floatX), borrow=True)
W_out = T.shared(np.random.uniform(-0.1, 0.1, (n_in, n_hid)).astype(T.config.floatX), borrow=True)

W_in_theta_update = T.shared(np.zeros((n_hid, n_in + 1), dtype=T.config.floatX), borrow=True)
W_rec_theta_update = T.shared(np.zeros((n_hid, n_hid), dtype=T.config.floatX), borrow=True)
W_out_theta_update = T.shared(np.zeros((n_in, n_hid), dtype=T.config.floatX), borrow=True)

x = TT.scalar('x', 'int32')
y = TT.scalar('y', 'int32')
lr = TT.scalar('lr', T.config.floatX)
h_init = np.asarray(np.ones(n_hid)) * 0.1

h = T.shared(h_init, borrow=True)

def forward_step(x_t):
    h_t = TT.nnet.sigmoid(TT.dot(W_rec, h) + W_in[:, x_t])
    y_t = TT.flatten(TT.nnet.softmax(TT.dot(W_out, h_t)),1)
#    h_t *= (x_t > 0)
    return [h_t, y_t]

[h_t_fb, y_t_fb] = forward_step(x)

def forward_batch_step(x_t, H_mask, H_tm1):
    H = TT.dot(W_rec,H_tm1) + W_in[:,x_t]
    H_t = TT.nnet.sigmoid(H)
    Y_t = TT.nnet.softmax(TT.transpose(TT.dot(W_out, H_t)))
    Y_t = -TT.log2(Y_t)
    Y_t = TT.dot(TT.transpose(Y_t), TT.diag(H_mask))
    return [H_t, Y_t]

[h_ts, y_predicted], _ = T.scan(forward_batch_step,
                         sequences=[X, X_MASK],
                         outputs_info=[H0, None]
                        )

#[h_ts_fb, y_predicted_fb], _ = T.scan(forward_step,
#                         sequences=[x],
#                         outputs_info=[h0, None]
#                        )

logprobs = y_predicted[TT.arange(Y.shape[0]), TT.transpose(Y), TT.reshape(TT.arange(n_minibatches),(n_minibatches,1))]
DENOM_th = TT.diag(1/TT.sum(logprobs>0, axis=1).astype('float32'))
cross_entropy = TT.sum(TT.dot(DENOM_th,logprobs)) / n_minibatches

#cross_entropy = -TT.mean(TT.log2(TT.nonzero_values(y_predicted[TT.arange(Y.shape[0]), TT.transpose(Y), TT.reshape(TT.arange(n_minibatches),(n_minibatches,1))])))
#cross_entropy_fb = -TT.mean(TT.log2(y_predicted_fb)[TT.arange(y.shape[0]), y])

cross_entropy_fb = -TT.log2(y_t_fb)[y]

params = [W_in, W_rec, W_out]
theta_updates = {W_in: W_in_theta_update, W_rec: W_rec_theta_update, W_out: W_out_theta_update}

g_params = []
for param in params:
    g_params.append(TT.grad(cross_entropy, param))
    T.pp(TT.grad(cross_entropy, param))

updates = []
for param, grad in zip(params, g_params):
    theta_update = theta_updates[param]
    upd = mom * theta_update - lr * grad
    updates.append((theta_updates[param], upd))
    updates.append((param, param + upd,))

train_fn = T.function(
    [X, Y, X_MASK, lr, mom, n_minibatches, H0],
    [cross_entropy],
    updates=updates
)

valid_fn = T.function(
    [x, y],
    cross_entropy_fb,
    updates=[(h, h_t_fb,)]
)

reset_fn = T.function([],[], updates={h: h_init})

#debug_fn = T.function([X, Y, X_MASK, n_minibatches, H0], [logprobs, DENOM_th, cross_entropy])

learning_rate = 0.1
valid_ent_prev = 100
decrease_started = False

print "Start!"

failed = 0

current_momentum = 0.995
for epoch in range(2000):
    if epoch > 10:
        current_momentum = 0.9
    position = 0
    i = 0
    while position < train_len:
        batch_matrix, masks_matrix, position, n_minibatches_real = get_minibatch_matrix(train_data, n_minibatches_num, position)
        train_ent = train_fn(batch_matrix[:-1,:], batch_matrix[1:,:], np.transpose(masks_matrix), learning_rate, current_momentum, n_minibatches_real, np.ones((n_hid,n_minibatches_real), T.config.floatX)*0.1)
        print "minibatch no. %s of shape %s by %s is processed. Av.entropy is %s" % (i, batch_matrix.shape[0], batch_matrix.shape[1], train_ent[0] )
        i += 1
    print "Training finished"
    word_cnt = 0
    reset_fn()
    valid_ent = 0
    for index in range(valid_len - 1):
        if valid_data[index + 1] != oov_code:
            ce_t = valid_fn(valid_data[index], valid_data[index + 1])
            valid_ent += ce_t
            word_cnt += 1
        else:
            valid_fn(valid_data[index], valid_data[index + 1]) #we do it just to switch the state. we don't actually predict+
    valid_ent /= word_cnt
    print "Epoch: %s  Valid entropy: %s" % (epoch,  valid_ent)

    if valid_ent > valid_ent_prev * 1.0002:
        failed += 1
        if failed == 2:
            #if not decrease_started:
               # print "Decrease started:"
               # decrease_started = True
            if learning_rate > 2 ** -7:
               learning_rate /= 2
               print learning_rate
            else:
                print "Finished"
                break
    else:
        failed = 0

    #f decrease_started:
     #  learning_rate /= 2

    valid_ent_prev = valid_ent

    if epoch == 1999:
        print "Maximum epoch reached"
