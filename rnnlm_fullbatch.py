import numpy as np
import theano as T
import theano.tensor as TT
import cPickle

np.set_printoptions(threshold=np.nan)

path = "Data/fake_corp.npz"
path_dic = "Data/fake_dict.npz"

#dictionary = np.load(path_dic)

print "Data loading"
data = np.load(path)
oov_code = data["oov"]
n_words = data["n_words"]

train_data = np.append([0], data['train_words']).astype('int32')
train_data = np.asarray([x for x in train_data if x != oov_code]).astype('int32')
train_len = train_data.shape[0]

valid_data = np.append([0], data["valid_words"]).astype('int32')
valid_data = np.asarray([x for x in valid_data if x != oov_code]).astype('int32')
valid_len = valid_data.shape[0]

n_in = n_words
n_hid = 10


x = TT.vector('x', 'int32')
y = TT.vector('y', 'int32')
lr = TT.scalar('lr', T.config.floatX)
mom = TT.scalar('mom', T.config.floatX)
h0 = TT.vector('h0', T.config.floatX) 
print "Compilation"

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
#
# W_in = T.shared(w_in_debug)
# W_rec = T.shared(w_rec_debug)
# W_out = T.shared(w_out_debug)

W_in = T.shared(np.random.uniform(-0.1, 0.1, (n_hid, n_in)).astype(T.config.floatX), borrow=True, name='W_in')
W_rec = T.shared(np.random.uniform(-0.1, 0.1, (n_hid, n_hid)).astype(T.config.floatX), borrow=True, name='W_rec')
W_out = T.shared(np.random.uniform(-0.1, 0.1, (n_in, n_hid)).astype(T.config.floatX), borrow=True, name='W_out')

def forward_step(x_t, h_tm1):
    h_t = TT.nnet.sigmoid(TT.dot(W_rec, h_tm1) + W_in[:, x_t])
    y_t = TT.flatten(TT.nnet.softmax(TT.dot(W_out, h_t)),1)
    return [h_t, y_t]

[h_ts, y_predicted], _ = T.scan(forward_step,
                         sequences=[x],
                         outputs_info=[h0, None]
                        )

cross_entropy = -TT.mean(TT.log2(y_predicted)[TT.arange(y.shape[0]), y])

params = [W_in, W_rec, W_out]

g_params = []
v_ts = {}
for param in params:
    init = np.zeros(param.get_value(borrow=True).shape, dtype=T.config.floatX)
    v_ts[param] = T.shared(init)
    g_params.append(TT.grad(cross_entropy, param))

updates = []
for param, grad in zip(params, g_params):
    upd = mom * v_ts[param] - lr * grad
    updates.append((param, param + upd ,))
    updates.append((v_ts[param], upd, ))

train_fn = T.function(
    [x, y, lr, mom],
    [cross_entropy, TT.grad(cross_entropy, W_rec)],
    updates=updates,
    givens=[(h0, np.ones(n_hid, T.config.floatX)*0.1)]
)

valid_fn = T.function(
    [x, y],
    cross_entropy,
    givens=[(h0, np.ones(n_hid, T.config.floatX)*0.1)]
)

learning_rate = 0.1
valid_ent_prev = 100
decrease_started = False

print "Start!"
final_momentum=0.995
initial_momentum=0.0
momentum_switchover=10

failed = 0
best_valid_cost = valid_ent_prev
best_params = [(param.name, param.get_value()) for param in params]
improvement_started = False

for epoch in range(20000):
    effective_momentum = final_momentum if epoch > momentum_switchover else initial_momentum
    train_ent, grads = train_fn(train_data[:-1], train_data[1:], learning_rate, effective_momentum)
    valid_ent = valid_fn(valid_data[:-1], valid_data[1:])

    print "Epoch: %s Train entropy: %s Valid entropy: %s LR: %s" % (epoch, train_ent, valid_ent, learning_rate)

    if valid_ent >= best_valid_cost:
        failed += 1
        if failed >= 50 or not improvement_started:
            if learning_rate > 2 ** -7:
                learning_rate /= 2
                print "Learning rate is decreased to %s: " % learning_rate
                bparams = dict(best_params)
                for param in params:
                    param.set_value(bparams[param.name])
                failed = 0
            else:
                cPickle.dump(best_params, open('model.pkl', 'w'))
                print "Finished. Best result: %s" % best_valid_cost
                break
    else:
        improvement_started = True
        failed = 0

    #f decrease_started:
     #  learning_rate /= 2

    valid_ent_prev = valid_ent

    if valid_ent < best_valid_cost:
        best_valid_cost = valid_ent
        best_params = [(param.name, param.get_value()) for param in params]

    if epoch == 19999:
        print "Maximum epoch reached"

    if epoch + 1 % 10 == 0:
        cPickle.dump(best_params, open('model.pkl', 'w'))