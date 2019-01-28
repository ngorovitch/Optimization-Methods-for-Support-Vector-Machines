# conda create -n cvxopt-env python=3.5 cvxopt numpy scipy matplotlib jupyter
# To activate this environment, use
#     $ conda activate cvxopt-env
# To deactivate an active environment, use
#     $ conda deactivate
import cvxopt
import numpy as np
#import quadprog
import time

#%% SVM classifier class

class SVMClassifier():
    """Container object for the model used for bynary classification using SVM technique."""

    def __init__(self, kernel_h_p, C, kernel):
        self.kernel_h_p = kernel_h_p     # kernel hyper parameter
        self.C = C                       # regularization parameter
        if kernel == 'polynomial' or kernel == 'RBF': self.kernel = kernel # chosen kernel function: between 'polynomial' and 'RBF'
        else: raise ValueError('Invalid input value for the parameter kernel')
        self.training_time = 0           # to save the time needed for training
        self.NbrIterations = 0           # to save the number of iterations needed for training

    # The Kernel function, to map the data into a higher dimension feature space where linear separability is possible
    def __kernel(self, X, Y):
        kernel_h_p = self.kernel_h_p
        kernel = self.kernel
        if kernel == 'polynomial':
            return (np.dot(X.T, Y) + 1) ** kernel_h_p
        elif kernel == 'RBF':
            expo = np.linalg.norm(X - Y) ** 2
            return np.exp(-kernel_h_p*expo)
        else:
            raise ValueError('Invalid input value for the parameter kernel')

    # The SVM thinks
    # This is the decision function of the SVM
    def think(self, x):
        result = self.bias
        for alpha_i, x_i, y_i in zip(self.support_multipliers, self.support_vectors, self.support_vector_labels):
            result += float(alpha_i) * float(y_i) * self.__kernel(x_i, x)
        return result

    # We train the SVM through an optimization proccess using the input data and solving the resulting QP problem.
    def fit(self, training_set_inputs, training_set_outputs, routine = 'quadprog', verbose = False):
        if verbose:
            print()
            print("Building the SVM model...")

        classes = set(list(np.asfarray(training_set_outputs).flatten()))
        if len(classes) < 3:
            # Reinitializing some attributes to 0
            self.training_time = 0
            self.NbrIterations = 0
            # Input data
            X = training_set_inputs
            # Output data
            Y = training_set_outputs
            N = X.shape[0]
            '''
            the matrix form of the soft margins dual SVM problem we wish to solve is the following:
                                    1
                        min Γ(λ) = --- * λ.T * Y * K(X, X.T) * Y * λ - 1.T * λ
                         λ          2
                 subject to                 Y.T * λ = 0
                                            λ >= 0
                                            λ <= C
            '''
            start_time = time.time()
            # let's compute the kernel matrix
            K = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    K[i, j] = self.__kernel(X[i], X[j])

            if routine == 'cvxopt':
                '''
                The routine offers us the following interface for solving quadratic optimization problems:
                                  1
                            min  --- * x.T * P * x + q.T * x
                             x    2
                     subject to          Gx <= h
                                         Ax = b

                Let's put the soft margin dual problem in this format in order to use the cvxopt routine to solve it.
                '''
                # mapping our quadratic dual problem to the stardard form of the "cvxopt" tool
                # for that we need to construct P, q, A, b, G, h matrices as follows
                P = cvxopt.matrix(Y * K * Y)
                q = cvxopt.matrix(-np.ones(N))
                G = cvxopt.matrix(np.vstack((np.diag(np.ones(N) * -1), np.identity(N))))
                h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * self.C)))
                A = cvxopt.matrix(Y.astype('d'), (1, N))
                b = cvxopt.matrix(0.0)

                if verbose: cvxopt.solvers.options['show_progress'] = True
                else: cvxopt.solvers.options['show_progress'] = False
                solution = cvxopt.solvers.qp(P, q, G, h, A, b)
                lambdas = np.array(solution['x'])
                self.NbrIterations = solution['iterations']

            elif routine == 'quadprog':
                '''
                The routine offers us the following interface for solving quadratic optimization problems:
                                  1
                            min  --- * x.T * G * x - a.T * x
                             x    2
                     subject to          C.T * x >= b, where the equality constraints are the first rows of
                                                       the matrix C (the number of rows is specified using
                                                       the parameter 'meq' of the routine).

                This routine is gradient based and computes the gradient autonomously.
                Let's put the soft margin dual problem in this format in order to use the cvxopt routine to solve it.
                '''
                G = (Y * K * Y) + np.eye(N)*(1e-3) # adding the identity matrix times a small constant to G to avoid
                                                    # the following error: ValueError: matrix G is not positive definite
                                                    # as suggested here: https://github.com/facebookresearch/GradientEpisodicMemory/issues/2
                a = np.ones(N).reshape(-1,)
                # equality constraints
                C_eq = np.array(Y.astype('d'), dtype=float).reshape(1, N)
                b_eq = np.array([0.0],dtype=float)
                # inequality constraints
                C_ineq = np.array(np.vstack((np.identity(N), np.diag(np.ones(N) * -1))), dtype=float)
                b_ineq = np.array(np.hstack((np.zeros(N), np.ones(N) * -self.C)),dtype=float)
                # put all together
                C = np.concatenate((C_eq, C_ineq)).T
                b = np.concatenate((b_eq, b_ineq)).reshape(-1,)

                solution = quadprog.solve_qp(G, a, C, b, meq = 1)
                lambdas = np.array(solution[0])
                self.NbrIterations = solution[3][0]

            else:
                raise ValueError("Invalid value for parameter routine. The value should be in {'cvxopt', 'quadprog'}")

            self.training_time = time.time() - start_time
            # let's compute the bias
            # A small threshold (e.g., 1e-5) is chosen to find the support vectors
            # (corresponding to non-zero Lagrange multipliers, by complementary slackness condition).
            support_vector_indices = (lambdas > 0).reshape(-1) # some small threshold
            self.support_multipliers = lambdas[support_vector_indices]
            self.support_vectors = X[support_vector_indices]
            self.support_vector_labels = Y[support_vector_indices]

            # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
            # bias = y_k - \sum z_i y_i  K(x_k, x_i)
            # Thus we can just predict an example with bias of zero, and
            # compute error.
            self.bias = 0.0
            bias = np.mean([y_k - np.sign(self.think(x_k)).item() for (y_k, x_k) in zip(self.support_vector_labels, self.support_vectors)])
            self.bias = bias

        else:
            '''
            We are dealing with a multi class classification problem
            '''
            raise ValueError("'training_set_outputs' have more than 2 distinct classes. Use MultiClassSVMClassifier for multi class classification.")

    # We train the SVM through a decomposition method (SMO)
    def fit_decomposition(self, training_set_inputs, training_set_outputs, max_iter = 100, verbose = False):
        if verbose:
            print()
            print("Building the stochastic SVM model...")

        classes = set(list(np.asfarray(training_set_outputs).flatten()))
        if len(classes) < 3:
            # Reinitializing some attributes to 0
            self.training_time = 0; self.NbrIterations = 0
            # Input and Output data
            X = training_set_inputs; Y = training_set_outputs
            # Data shape
            N, n = X.shape[0], X.shape[1]

            '''We are implementing the SMO decomposition algorithm'''

            # Data.
            eps  = 10 ** -12         # a threshold to build the index sets because hardly the lagrange multiplier will equals exactly 0 or C
            alpha = np.ones(N)*1e-3   # the starting point α_0 = 0
            e = np.ones((N, 1))
            gradient = -e            # the initial gradient which does not require any element of the kernel matrix
            # Inizialization.
            k = 0                    # iterations counter
            f_primal_prev = 0
            Q = np.dot(np.dot(Y, Y.T), np.dot(X, X.T))
            #Q = np.dot(np.dot(Y, self.__rbf_kernel(X, X, self.kernel_h_p)),Y)
            G = np.vstack((np.diag(np.ones(2) * -1), np.identity(2)))
            h  = np.concatenate([np.zeros((2, 1)), np.full((2, 1), self.C)], 0)

            start_time = time.time()
            while True:
                alpha_prev = np.copy(alpha)
                '''
                Step 1. select i € R(α_k), j € S(α_k), such that:
                            gradient_f(α_k).T * d_i,j < 0, d_i,j is the descent direction and gradient_f(α_k).T
                            is the transposition of the gradient vector
                '''
                R, S = self.__get_index_sets(alpha, self.C, Y, 1e-3)
                i, j = self.__get_working_set(R, S, gradient, Y, N)
                W = [i, j]
                '''
                Step 2. Compute a solution α∗ = (α_i∗, α_j∗).T
                '''
                P_sub, alpha_sub, e_sub, Y_train_sub, Q_cols = self.__generate_subset(Q, X, Y, alpha, W, N, n)
                solution, nbr_iters, f_primal = self.__optimize_dual(P_sub, e_sub, G, Y_train_sub, 0, h)
                '''
                Step 3. Compute α_k+1
                '''
                alpha[i] = solution[0]
                alpha[j] = solution[1]
                self.NbrIterations += nbr_iters
                '''
                Step 4. Update the gradient
                '''
                gradient = gradient + np.reshape(np.sum(Q_cols, axis=0)*(alpha-alpha_prev), (N, 1))
                '''
                Step 5. set k = k + 1
                '''
                k += 1
                '''
                Check convergence
                '''
                if (k >= max_iter):
                    break
                if(f_primal == f_primal_prev):
                    print("Solution found")
                    break

                f_primal_prev = f_primal
            self.training_time = time.time() - start_time
            self.m_a = j
            self.M_a = i
            # let's compute the bias
            self.support_vectors, self.support_vector_labels, self.support_multipliers, N_sup_vec = self.__get_support_vectors(X, Y, alpha, n, eps)
            b_star = self.__generate_bias(self.support_multipliers, self.support_vectors, self.support_vector_labels, N_sup_vec)
            self.bias = b_star
        else:
            '''
            We are dealing with a multi class classification problem
            '''
            raise ValueError("'training_set_outputs' have more than 2 distinct classes. Use MultiClassSVMClassifier for multi class classification.")

    # We train the SVM through a most violating pair(MVP) decomposition method
    def fit_MVP(self, training_set_inputs, training_set_outputs, max_iter = 100, verbose = False):
        if verbose:
            print()
            print("Building the stochastic SVM model...")

        classes = set(list(np.asfarray(training_set_outputs).flatten()))
        if len(classes) < 3:
            # Reinitializing some attributes to 0
            self.training_time = 0; self.NbrIterations = 0
            # Input and Output data
            X = training_set_inputs; Y = training_set_outputs
            # Data shape
            N, n = X.shape[0], X.shape[1]
            Y_extend = np.eye(N) * Y

            '''We are implementing a MVP decomposition algorithm'''

            # Data.
            eps  = 10 ** -12         # a threshold to build the index sets because hardly the lagrange multiplier will equals exactly 0 or C
            alpha = np.ones(N)*1e-3   # the starting point α_0 = 0
            e = np.ones((N, 1))
            gradient = -e            # the initial gradient which does not require any element of the kernel matrix
            # Inizialization.
            k = 0                    # iterations counter
            if self.kernel == 'RBF':
                Q = np.dot(np.dot(Y_extend, self.__rbf_kernel(X, X, self.kernel_h_p)), Y_extend)
            else:
                Q = np.dot(np.dot(Y_extend, self.__polynomial_kernel(X, X, self.kernel_h_p)), Y_extend)
            grad_y = np.zeros((2,N))

            start_time = time.time()
            while True:
                di = np.zeros(N)

                '''
                Step 1. select i € R(α_k), j € S(α_k), such that:
                            gradient_f(α_k).T * d_i,j < 0, d_i,j is the descent direction and gradient_f(α_k).T
                            is the transposition of the gradient vector
                '''
                grad_y[0,:] = np.reshape(self.__get_grad(gradient, Y), (N))
                grad_y[1,:] = np.array(np.arange(N))
                R, S = self.__get_index_sets(alpha, self.C, Y, 1e-3)
                grad_R = np.zeros((2,len(R)))
                grad_S = np.zeros((2,len(S)))
                r=0
                s=0
                for i in range(len(grad_y[1])):
                    if   i in R:
                        grad_R[:,r]=grad_y[:,i]
                        r+=1
                    elif i in S:
                        grad_S[:,s]=grad_y[:,i]
                        s+=1

                grad_R_sorted = grad_R[:,grad_R[0,:].argsort()]
                grad_S_sorted = grad_S[:,grad_S[0,:].argsort()]

                mi = grad_S_sorted[1][0]
                Ma = grad_R_sorted[1][len(R)-1]
                I = int(grad_R_sorted[1][len(R)-1])
                J = int(grad_S_sorted[1][0])
                '''
                Step 2. Compute a solution α∗ = (α_i∗, α_j∗).T
                '''
                obj_f = self.__obj_fun(alpha, Q, e)
                if(obj_f < 0 and mi - Ma < 10**-2):
                    print("Solution Found")
                    break

                # Set up the decent directions
                # Check theorem 5.6
                # Algorithm 5.A.1
                di[I] =   Y[I]
                di[J] = - Y[J]


                d_grad = np.dot(gradient.T, di)
                if d_grad > 0: di = -di
                if np.dot(np.dot(di.T, Q), di) > 0:
                    d = -np.dot(gradient.T, di) / np.dot(np.dot(di.T, Q), di)
                else: d = float("+inf")
                d_bar = self.__get_di(di[I], di[J], alpha[I], alpha[J], self.C)
                d_star = min(d, d_bar)

                # get new alpha[J]
                alpha[J] = alpha[J] + (d_star * di[J])
                if(alpha[J] < eps): alpha[J] = eps
                elif(alpha[J] > self.C - eps): alpha[J] = self.C

                # get new alpha[I]
                alpha[I] = alpha[I] + (d_star * di[I])
                if(alpha[I]< eps): alpha[I] = eps
                elif(alpha[I] > self.C - eps): alpha[I] = self.C
                '''
                Step 3. Update the gradient
                '''
                gradient = gradient + d_star * np.reshape((Q[:, I] + Q[:, J]), (N, 1))
                '''
                Step 5. set k = k + 1
                '''
                k += 1
                if (k >= max_iter):
                    break
            self.NbrIterations = k
            self.m_a = mi
            self.M_a = Ma
            self.training_time = time.time() - start_time

            # let's compute the bias
            self.support_vectors, self.support_vector_labels, self.support_multipliers, N_sup_vec = self.__get_support_vectors(X, Y, alpha, n, eps)
            b_star = self.__generate_bias(self.support_multipliers, self.support_vectors, self.support_vector_labels, N_sup_vec)
            self.bias = b_star
        else:
            '''
            We are dealing with a multi class classification problem
            '''
            raise ValueError("'training_set_outputs' have more than 2 distinct classes. Use MultiClassSVMClassifier for multi class classification.")

    # refer to 5A 115 for function
    def __get_di(self, di, dj, alphai, alphaj, c):
        if (di>0 and dj>0):
            t= min(c-alphai,c-alphaj)
        elif (di<0 and dj<0):
            t= min(alphai,alphaj)
        elif (di>0 and dj<0):
            t= min(c-alphai,alphaj)
        elif (di<0 and dj>0):
            t= min(alphai,c-alphaj)
        return t

    def __obj_fun(self, alpha, Q, e):
        return (np.dot(np.dot(alpha.T,Q),alpha)-np.dot(e.T,alpha))/2

    def __get_support_vectors(self, x, y, alpha, cols, treshold):
        support_vectors_num = 0
        # get the support vectors indices
        # Setting all non support vec to 0 and returning the rest
        for a in range(len(alpha)):
            if (alpha[a] < treshold):
                alpha[a] = 0
            else:
                 support_vectors_num += 1
        # initialization of fuport vect matrices
        X = np.zeros(( support_vectors_num, cols))
        Y = np.zeros( support_vectors_num)
        A = np.zeros( support_vectors_num)
        pos = 0

        for i in range(len(alpha)):
            if (alpha[i] != 0):
                X[pos, :] = x[i, :]
                Y[pos] = y[i]
                A[pos] = alpha[i]
                pos += 1
        return X, Y, A, support_vectors_num

    # get the best bias
    def __generate_bias(self, alpha, x, y, N):
        if self.kernel == 'polynomial':
            return (np.sum((1 - np.dot(alpha * y, self.__polynomial_kernel(x, x, self.kernel_h_p))),0) / float(N))
        else:
            return (np.sum((1 - np.dot(alpha * y, self.__rbf_kernel(x, x, self.kernel_h_p))),0) / float(N))

    def __optimize_dual(self, P, e, G, Y, b, cons):

        Q = cvxopt.matrix(P)
        e = cvxopt.matrix(e)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(cons)
        A = cvxopt.matrix(np.array([Y]))
        b = cvxopt.matrix(np.full((1, 1), float(b)))
        cvxopt.solvers.options['maxiters'] = 1000
        res = cvxopt.solvers.qp(Q, e, G, h, A, b)
        return np.array(res['x']).flatten(), res['iterations'], res['primal objective']

    def __generate_subset(self, P, x, y, alpha, W, N, n):
        # Size of matrix W
        working_set_size = len(W)
        # initialize de matrices
        x_sub = np.zeros((working_set_size, n))
        y_sub, a_sub = np.zeros(working_set_size), np.zeros(working_set_size)

        a_rows = np.zeros(N - len(W))
        temp= np.zeros((N, len(W)))

        #initialize Q Matrix
        Q_rows = np.zeros(((N-len(W)), len(W)))
        Q_cols = np.zeros((len(W), len(alpha)))

        for i in range(len(W)):
            x_sub[i] = x[int(W[i])]
            y_sub[i] = y[int(W[i])]
            a_sub[i], temp[:,i] = alpha[int(W[i])], P[:, int(W[i])]

        c, d = 0,0
        for i in range(N):
            if i not in W:
                Q_rows[c,:], a_rows[c] = temp[i,:], alpha[i]
                c+=1
            else:
                Q_cols[d,:] = P[:, i]
                d+=1
        if self.kernel == 'polynomial':
            Q_sub = np.dot((np.dot(np.diag(y_sub), self.__polynomial_kernel(x_sub, x_sub, self.kernel_h_p))), np.diag(y_sub))
        else:
            Q_sub = np.dot((np.dot(np.diag(y_sub), self.__rbf_kernel(x_sub, x_sub, self.kernel_h_p))), np.diag(y_sub))

        e_sub = np.dot(a_rows, Q_rows) - np.ones(len(W))

        return Q_sub, a_sub, e_sub, y_sub, Q_cols

    # Polynomial kernel
    def __polynomial_kernel(self, x, y, kernel_h_p):
        return (np.matmul(x, y.T) + 1) ** kernel_h_p

    # RBF kernel
    def __rbf_kernel(self, x, y, gamma):
        return np.exp(-gamma * (np.sum(x ** 2, axis=-1)[:, None] +
                                  np.sum(y ** 2, axis=-1)[None, :] -
                                  2 * np.dot(x, y.T)))

    def __get_index_sets(self, alpha, C, Y, eps):

        L_pos, L_neg, U_pos, U_neg, mid = [],[],[],[],[]
        for a in range(len(alpha)):
            if  (alpha[a] <= eps and Y[a] >  0): L_pos.append(a)
            elif(alpha[a] >= C - eps and Y[a] > 0): U_pos.append(a)
            elif(alpha[a] <= eps and Y[a] < 0): L_neg.append(a)
            elif(alpha[a] >= C - eps and Y[a] < 0): U_neg.append(a)
            else: mid.append(a)

        R = L_pos + U_neg + mid
        S = L_neg + U_pos + mid

        return R, S

    def __get_working_set(self, R, S, gradient, Y, N):
        grad_y=np.zeros((2,N))
        grad_y[0,:] = np.reshape(self.__get_grad(gradient, Y), (N))
        grad_y[1,:] = np.array(np.arange(N))
        #initialize the gradient of the working sets
        grad_R = np.zeros((2,len(R)))
        grad_S = np.zeros((2,len(S)))
        r=0
        s=0
        for i in range(len(grad_y[1])):
            if i in R:
                grad_R[:, r] = grad_y[:, i]
                r+=1
            elif i in S:
                grad_S[:, s] = grad_y[:, i]
                s+=1

        grad_R_sorted = grad_R[:, grad_R[0, :].argsort()]
        grad_S_sorted = grad_S[:, grad_S[0, :].argsort()]

        index_mi = grad_S_sorted[1][:1]
        index_Ma = grad_R_sorted[1][len(R)-1:]
        return int(index_Ma), int(index_mi)

    # Calculate the gradient
    def __get_grad(self, grad, val):
        return -np.reshape(val, (len(val), 1))*grad

    def predict(self, test_set_inputs):
        # test data
        X = test_set_inputs
        # number of records in the input test data
        P = len(test_set_inputs)
        y_hat = []
        for iteration in range(P):
            y_hat.append(np.sign(self.think(X[iteration])))
        return np.asarray(y_hat, dtype=np.int64).reshape(P, 1)

    # We compute the error given the the estimated outputs and the true outputs values
    def get_error(self, y_hat, y):
        P = len(y_hat)
        sum = 0
        for iteration in range(P):
            if str(y[iteration][0]) == str(y_hat[iteration][0]):
                sum += 1
        return sum/P

#%% Multiclass SVM classifier class

class MultiClassSVMClassifier():
    def __init__(self, kernel_h_p, C, kernel):
        self.kernel_h_p = kernel_h_p
        self.C = C
        self.kernel = kernel
        self.training_time = 0


    def fit(self, training_set_inputs, training_set_outputs, routine = 'quadprog', verbose = False):
        print()
        print("Building the multi class SVM model...")
        classes = set(list(np.asfarray(training_set_outputs).flatten()))
        if len(classes) > 2:
            '''
            We are dealing with multiclass classification we are going to proceed as follows:
                1- Build a classifier for each class, where the training set consists of the set of documents in the class (positive labels) and its complement (negative labels).
                2- Given the test document, apply each classifier separately.
                3- Assign the document to the class with
                    - the maximum score,
            '''
            models = {}
            Y = training_set_outputs.copy()
            for c in classes:
                print()
                print("     Training the SVM model for class: "+str(int(c))+" ...")
                print()
                other_classes = (Y != c).reshape(-1)
                this_class = (Y == c).reshape(-1)
                Y[other_classes] = -1
                Y[this_class] = 1
                svm = SVMClassifier(kernel_h_p = self.kernel_h_p, C = self.C, kernel = self.kernel)
                svm.fit(training_set_inputs, Y, routine = routine, verbose = verbose)
                models[str(int(c))] = svm
                Y = training_set_outputs.copy()

            self.models = models
            self.training_time = sum([x.training_time for x in models.values()])
            self.NbrIterations = sum([x.NbrIterations for x in models.values()])
        else:
            '''
            We are dealing with a binary classification problem
            '''
            raise ValueError("'training_set_outputs' have less than 3 distinct classes. Use SVMClassifier for binary classification.")

    def predict(self, test_set_inputs):
        # test data
        X = test_set_inputs
        # number of records in the input test data
        P = len(test_set_inputs)
        y_hat = []
        # for each data sample
        for iteration in range(P):
            results = []
            # predict using each of the k (k = number of classes) models
            for model in self.models.values():
                results.append(float(model.think(X[iteration])))
            idx_best_model = results.index(max(results))
            # if the max value of the decision function is negative we asign the data
            # sample to a k+1 class 'NA' suggesting that the data sample cannot be assign to none of the
            # k classes that were available on the training dataset
            if np.sign(max(results)) == -1:
                y_hat.append(-1)
            else:
                y_hat.append([*self.models][idx_best_model])
        return np.asarray(y_hat).reshape(P, 1)

    # We compute the error given the the estimated outputs and the true outputs values
    def get_error(self, y_hat, y):
        P = len(y_hat)
        sum = 0
        for iteration in range(P):
            if str(y[iteration][0]) == str(y_hat[iteration][0]):
                sum += 1
        return sum/P