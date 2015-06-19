from random import randrange
class BaseDataset:
    def __init__(self, flatten, training_set_size=100, height=100, width=100):
        self._training_set_size = training_set_size
        self._height = height
        self._width = width
        self._flatten = flatten
        self._X = []
        self._y = []
        self._valid_X = []
        self._valid_y = []

    def read_dataset(self):
        self._X, self._y = self.read_training_set()
        self._valid_X, self._valid_y = self.read_validation_set()

    def _read_image(self, filepath):
        image = misc.imread(filepath)

        grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
        # get row number
        for rownum in range(len(image)):
               for colnum in range(len(image[rownum])):
                         grey[rownum][colnum] = np.average(image[rownum][colnum])

        grey = grey/255. #normalize
        grey = grey -grey.mean()
        return grey

    def _return_training_set(self, X, y):
        y = np.array(y)
        X = np.array(X)
        X = X.astype(np.float32)
        if not self._flatten:
            X = X.reshape(-1, 1, self._height, self._width)
        print X.__class__, y.__class__
        return X,y

    def cross_validation(self, number_of_folds):
        self._X, self._y = self.read_training_set()
        print "length" , len(self._X)
        fold_sets = [randrange(number_of_folds)
                     for x
                     in xrange(len(self._X))]
        for fold in xrange(number_of_folds):
            X, y = self._extract_fold_set(fold_sets, fold, "training")
            X,y = self._return_training_set(X,y)
            X_validation, y_validation = self._extract_fold_set(fold_sets, fold, "validation")
            X_validation,y_validation = self._return_training_set(X_validation,y_validation)
            yield X, y, X_validation, y_validation

    def _extract_fold_set(self, fold_sets, fold, set_type):
        if set_type == "training":
            eq = lambda x,y: x==y
        elif set_type == "validation":
            eq = lambda x,y: x!=y

        print len(fold_sets), len(self._X), self._training_set_size, len(fold_sets)

        return [self._X[i]
                for i
                in xrange(self._training_set_size)
                if eq(fold_sets[i], fold)], \
               [self._y[i]
                for i
                in xrange(self._training_set_size)
                if eq(fold_sets[i],fold)]

    def _extract_validation_training_set(self,
                                         fold_sets,
                                         fold):

            print len(self._X), len(fold_sets)
            return [self._X[i]
                    for i
                    in xrange(self._training_set_size)
                    if fold_sets[i] != fold], \
                   [self._y[i]
                    for i
                    in xrange(self._training_set_size)
                    if fold_sets[i] != fold]

    def _extract_validation_validation_set(self, fold_sets, fold):
            return [self._X[i]
                    for i
                    in xrange(self._training_set_size)
                    if fold_sets[i] == fold], \
                   [self._y[i]
                    for i
                    in xrange(self._training_set_size)
                    if fold_sets[i] == fold]
