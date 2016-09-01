import numpy as np
import theano as theano
import theano.tensor as T

class RNNTheano:

    def __init__(self, w_dim, h_dim=100, bptt_max=4):

        ## Set self parameters.
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.bptt_max = bptt_max

        ## Randomly initialize network parameters. Set to uniform between
        ## [-1/n, 1/n], where 'n' is the size of incoming connections.
        E = np.random.uniform(-np.sqrt(1./w_dim), np.sqrt(1./w_dim), (h_dim, w_dim))
        U = np.random.uniform(-np.sqrt(1./h_dim), np.sqrt(1./h_dim), (8, h_dim, h_dim))
        W = np.random.uniform(-np.sqrt(1./h_dim), np.sqrt(1./h_dim), (8, h_dim, h_dim))
        V = np.random.uniform(-np.sqrt(1./h_dim), np.sqrt(1./h_dim), (w_dim, h_dim))
        b = np.zeros((6, h_dim))
        c = np.zeros(w_dim)

        ## Assign to self as theano shared variables.
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

        ## Create cache variables.
        self.E_ = theano.shared(name='E_', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.U_ = theano.shared(name='U_', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.W_ = theano.shared(name='W_', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.V_ = theano.shared(name='V_', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.b_ = theano.shared(name='b_', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.c_ = theano.shared(name='c_', value=np.zeros(c.shape).astype(theano.config.floatX))

        ## Build Model.
        self.theano = {}
        self.build_model()

    def build_model(self):

        ## Define parameters.
        x = T.ivector('x')
        y = T.ivector('y')
        E, U, W, V, b, c = self.E, self.U, self.W, self.V, self.b, self.c

        def fwd_step(x_t, s1_t_prev, s2_t_prev, c1_t_prev, c2_t_prev):
            ## Retrieve word embeddings.
            x_e = E[:,x_t]

            ## LSTM Layer 1.
            i1_t = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s1_t_prev) + b[0])
            f1_t = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s1_t_prev) + b[1])
            o1_t = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s1_t_prev) + b[2])
            g1_t = T.tanh(U[3].dot(x_e) + W[3].dot(s1_t_prev) + b[3])
            c1_t = c1_t_prev * f1_t + g1_t * i1_t
            s1_t = T.tanh(c1_t) + o1_t

            ## LSTM Layer 2.
            i2_t = T.nnet.hard_sigmoid(U[0].dot(s1_t) + W[0].dot(s2_t_prev) + b[0])
            f2_t = T.nnet.hard_sigmoid(U[1].dot(s1_t) + W[1].dot(s2_t_prev) + b[1])
            o2_t = T.nnet.hard_sigmoid(U[2].dot(s1_t) + W[2].dot(s2_t_prev) + b[2])
            g2_t = T.tanh(U[3].dot(s1_t) + W[3].dot(s2_t_prev) + b[3])
            c2_t = c2_t_prev * f2_t + g2_t * i2_t
            s2_t = T.tanh(c2_t) + o2_t

            ## Send output to softmax.
            o_t = T.nnet.softmax(V.dot(s2_t) + c)[0]

            return [o_t, s1_t, s2_t, c1_t, c2_t]

        ## Iterate over all observations
        [o, s, s2, c1, c2], updates = theano.scan(fwd_step,
                                      sequences=x,
                                      outputs_info=[None,
                                                    dict(initial = T.zeros(self.h_dim)),
                                                    dict(initial = T.zeros(self.h_dim)),
                                                    dict(initial = T.zeros(self.h_dim)),
                                                    dict(initial = T.zeros(self.h_dim))],
                                      #non_sequences=[E,U,W,V],
                                      #strict=True,
                                      truncate_gradient=self.bptt_max)

        ## Prediction and loss.
        predict = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        ## Define gradients.
        dE = T.grad(o_error, E)
        dU = T.grad(o_error, U)
        dW = T.grad(o_error, W)
        dV = T.grad(o_error, V)
        db = T.grad(o_error, b)
        dc = T.grad(o_error, c)

        ## Convert graph into actual functions.
        self.fwd_prop = theano.function([x], o)
        self.predict = theano.function([x], predict)
        self.ce_error = theano.function([x,y], o_error)
        self.bptt = theano.function([x,y], [dE, dU, dW, dV, db, dc])

        ## SGD variables.
        l_rate = T.scalar('l_rate')
        decay = T.scalar('decay')

        # Following the rmsprop approach:
        E_ = decay * self.E_ + (1 - decay) * dE ** 2
        U_ = decay * self.U_ + (1 - decay) * dU ** 2
        W_ = decay * self.W_ + (1 - decay) * dW ** 2
        V_ = decay * self.V_ + (1 - decay) * dV ** 2
        b_ = decay * self.b_ + (1 - decay) * db ** 2
        c_ = decay * self.c_ + (1 - decay) * dc ** 2

        ## SGD.
        self.sgd_step = theano.function([x,y,l_rate, theano.In(decay, value=0.9)], [],
                                        updates = [(E, E - l_rate * dE / T.sqrt(E_ + 1e-6)),
                                                   (U, U - l_rate * dU / T.sqrt(U_ + 1e-6)),
                                                   (W, W - l_rate * dW / T.sqrt(W_ + 1e-6)),
                                                   (V, V - l_rate * dV / T.sqrt(V_ + 1e-6)),
                                                   (b, b - l_rate * db / T.sqrt(b_ + 1e-6)),
                                                   (c, c - l_rate * dc / T.sqrt(c_ + 1e-6)),
                                                   (self.E_, E_),
                                                   (self.U_, U_),
                                                   (self.W_, W_),
                                                   (self.V_, V_),
                                                   (self.b_, b_),
                                                   (self.c_, c_)])

    ## Total loss function...
    def total_loss_function(self, X, Y):
        return np.sum([self.ce_error(x, y) for x,y in zip(X,Y)])

    ## Loss by word dimension.
    def loss_function(self, X, Y):
        n = np.sum([len(y) for y in Y])
        return self.total_loss_function(X, Y) / float(n)
