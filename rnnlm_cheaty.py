#!/usr/bin/python

import numpy as np
import theano as T
import theano.tensor as TT
import datetime

np.set_printoptions(threshold=np.nan)

path = "/home/mkudinov/Data/mikolov_corp.npz"
path_dic = "/home/mkudinov/Data/mikolov_dict.npz"

#dictionary = np.load(path_dic)

print "Data loading"
data = np.load(path)
oov_code = data["oov"]
n_words = data["n_words"]

# train_data = np.append([0], data['train_words']).astype('int32')
# train_data = np.asarray([x for x in train_data if x != oov_code]).astype('int32')
# train_len = train_data.shape[0]
#
# valid_data = np.append([0], data["valid_words"]).astype('int32')
# valid_data = np.asarray([x for x in valid_data if x != oov_code]).astype('int32')
# valid_len = valid_data.shape[0]

train_data = data['train_words'].astype(np.int32)
train_len = train_data.shape[0]
valid_data = data['valid_words'].astype(np.int32)
valid_len = valid_data.shape[0]

n_in = n_words
n_hid = 15

print "Compilation"

x = TT.scalar('x', 'int32')
y = TT.scalar('y', 'int32')
lr = TT.scalar('lr', T.config.floatX)
h_init = np.asarray(np.ones(n_hid)) * 0.1

# w_rec_debug= np.asarray([[0.0320476,0.066083,0.153841,-0.147026,-0.063141,-0.162517,-0.034802,0.0929369,-0.00842578,0.051775],
#                         [0.00901554,-0.111701,-0.0393406,-0.191836,0.122224,0.0314692,0.0295982,-0.0522779,0.160056,0.048756],
#                         [0.0860805,0.140482,0.144441,0.0794844,0.0218575,0.103608,0.0908192,0.0373316,-0.10388,-0.0959577],
#                         [0.0634808,-0.120983,-0.103742,0.0302188,-0.103259,0.00483933,-0.0571858,0.171325,-0.00146527,-0.153331,],
#                         [-0.0560495,0.0844793,-0.100088,-0.112521,0.124882,-0.00179792,-0.0520793,0.0184794,0.189363,-0.017572,],
#                         [-0.0343786,0.0591911,0.00544616,-0.0506752,-0.173127,0.00157214,0.0524387,-0.0242688,0.101668,0.0269906],
#                         [-0.0600803,-0.0846658,0.101512,-0.0718312,0.0372523,0.0727131,0.0190305,-0.123884,0.0314498,-0.0900866],
#                         [-0.0305857,0.162813,0.017018,0.0687782,0.244248,0.00904942,0.122373,0.0896645,-0.0397506,0.0610006],
#                         [-0.075115,-0.148395,0.023429,0.173482,0.0298347,0.0324082,-0.203464,-0.0762692,-0.157839,0.0890785],
#                         [0.0627398,-0.107054,0.170572,-0.0256244,-0.0601043,-0.0634536,0.0271406,0.128696,0.00999162,-0.184725]], T.config.floatX)
#
# w_in_debug = np.asarray([[0.103534,0.0815278,-0.0237546],
#                          [0.113603,0.0150146,0.115842],
#                          [0.0363778,-0.00276921,-0.0222987],
#                          [-0.00141754,-0.155062,0.236046],
#                          [-0.152804,0.129715,-0.0580521],
#                          [0.201284,0.236749,0.226155],
#                          [-0.143462,0.0331233,0.0280724],
#                          [0.00970916,-0.0212423,-0.0644614],
#                          [-0.10651,0.0101574,0.0867328],
#                          [0.0371972,0.00554711,-0.0297553]], T.config.floatX)
#
# w_out_debug = np.asarray([[0.0535957,-0.252673,0.0861321,-0.0697384,0.0407159,0.0109851,-0.0487732,0.0481737,-0.0732891,-0.0615571],[0.00679874,0.00525988,0.0636611,-0.0855893,0.0540449,-0.0937056,0.01332,-0.0357996,-0.103867,-0.0853364],[-0.109962,-0.0150034,-0.00751596,0.183073,0.0311375,-0.0234252,0.0559308,-0.00365388,0.026517,-0.143565]], T.config.floatX)

# W_in = T.shared(w_in_debug)
# W_rec = T.shared(w_rec_debug)
# W_out = T.shared(w_out_debug)

W_init = np.random.uniform(-0.1, 0.1, (n_hid, n_in + 1))
W_init[:,oov_code] = np.asarray([0] * n_hid)
W_in = T.shared(np.random.uniform(-0.1, 0.1, (n_hid, n_in + 1)).astype(T.config.floatX))
W_rec = T.shared(np.random.uniform(-0.1, 0.1, (n_hid, n_hid)).astype(T.config.floatX))
W_out = T.shared(np.random.uniform(-0.1, 0.1, (n_in, n_hid)).astype(T.config.floatX))

h = T.shared(h_init, borrow=True)

def forward_step(x_t):
    h_t = TT.nnet.sigmoid(TT.dot(W_rec, h) + W_in[:, x_t])
    y_t = TT.flatten(TT.nnet.softmax(TT.dot(W_out, h_t)), 1)
    return [h_t, y_t]

[h_t, y_t] = forward_step(x)

lp = -TT.log2(y_t)[y]

params = [W_in, W_rec, W_out]

g_params = []
for param in params:
    g_params.append(TT.grad(lp, param))

updates = []
for param, grad in zip(params, g_params):
    updates.append((param, param - grad * lr ,))

updates.append((h, h_t,))

train_fn = T.function(
    [x, y, lr],
    lp,
    updates=updates
)

valid_fn = T.function(
    [x, y],
    lp,
    updates=[(h, h_t,)]
)

reset_fn = T.function([],[], updates={h: h_init})

learning_rate = 0.1
valid_ent_prev = 100
decrease_started = False

print "Run"

time_start = datetime.datetime.now()

epoch = 0

while True:
    train_ent = 0
    valid_ent = 0

    reset_fn()
    for index in range(train_len - 1):
        ce_t = train_fn(train_data[index], train_data[index + 1], learning_rate)
        train_ent += ce_t
    train_ent /= train_len
#    print "Train entropy: %s" % train_ent

    word_cnt = 0
    reset_fn()
    for index in range(valid_len - 1):
        if valid_data[index + 1] != oov_code:
            ce_t = valid_fn(valid_data[index], valid_data[index + 1])
            valid_ent += ce_t
            word_cnt += 1
        else:
            valid_fn(valid_data[index], valid_data[index + 1]) #we do it just to switch the state. we don't actually predict+
    valid_ent /= word_cnt
    print "Epoch: %s Train entropy: %s Valid entropy: %s LR: %s" % (epoch, train_ent, valid_ent, learning_rate)


    if valid_ent * 1.003 >= valid_ent_prev:
        if not decrease_started:
            print "Decrease started:"
            decrease_started = True
        else:
            print "Finished"
            break

    if decrease_started:
        learning_rate /= 2

    valid_ent_prev = valid_ent
    epoch+=1

print "Running time: %s seconds" % ( datetime.datetime.now() - time_start)
