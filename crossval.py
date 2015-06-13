from util import quadratic_weighted_kappa

def crossvalidation(dataset, network_class, number_of_folds):
    scores = []
    for X,y, validation_X, validation_y in dataset.cross_validation(number_of_folds):
        network = network_class()
        network.train(X,y)

        results = network.predict(validation_X)
        results_scalar += [result.argmax() for result in results]
        target_scalar += [target.argmax() for target in validation_y]
        kappa = quadratic_weighted_kappa(results_scalar, target_scalar)
        scores.append(kappa)
        log("fold kappa: %s" % (kappa))
    average = sum(scores) / len(scores)
    log("kappa average = %s" % average)
    return average, scores
