class BaseDataset:
    def __init__(self, flatten, training_set_size=100, height=100, width=100):
        self._training_set_size = training_set_size
        self._height = height
        self._width = width
        self._flatten = flatten
        self._X = []
        self._y = []

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
        return X,y

    def read_training_set(self):
        print "implement this function"

    def cross_validation(self, number_of_folds):
        fold_sets = [randrange(number_of_folds)
                     for x
                     in xrange(len(self._X))]
        for fold in xrange(number_of_folds):
            X, y = self._extract_validation_training_set(fold_sets, fold)
            X,y = self._return_training_set(X,y)
            X_validation, y_validation = self._extract_validation_validation_set(fold_sets, fold)
            yield X, y, X_validation, y_validation

    def _extract_validation_training_set(self,
                                         fold_sets,
                                         fold):
            return [self._X[i]
                    for i
                    in xrange(len(self._X))
                    if fold_sets[i] != fold],
                   [self._y[i]
                    for i
                    in xrange(len(self._y))
                    if fold_sets[i] != fold]

    def _extract_validation_validation_set(self, fold_sets, fold):
            return [self._X[i]
                    for i
                    in xrange(len(self._X))
                    if fold_sets[i] == fold],
                   [self._y[i]
                    for i
                    in xrange(len(self._y))
                    if fold_sets[i] == fold]
