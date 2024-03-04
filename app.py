import parsing
import classifiers
import dataTreatment
import numpy as np

separator = "\n ============================================================================== \n"

def main():
    args = parsing.parse()
    if args.verbose:
        print("arguments parsed: ", args, separator)

    # DATA TREATMENT
    (y, unique_values) = dataTreatment.y(args.y)
    if args.verbose:
        print("finished reading class file",  separator)
    x = dataTreatment.x(args.x)
    if args.verbose:
        print("finished reading fasta file", separator)
    kmers = dataTreatment.generate_kmers(args.kmer)
    if args.verbose:
        print("finished generating kmers", separator)
    (df, index) = dataTreatment.matrice(kmers, x)
    if args.verbose:
        print("finished creating matrix", separator)
    x_matrix = dataTreatment.count_kmers(kmers, x, df, index)
    if args.verbose:
        print("finished counting kmers", x_matrix, x_matrix.shape ,separator)

    # CLASSIFICATION
    (x_train, x_test, y_train, y_test) = classifiers.split_data(x_matrix, y)
    if args.verbose:
        print("finished splitting data", "\n x_train shape: ", x_train.shape, 
              "\n x_test shape: ", x_test.shape, "\n y_train shape: ", y_train.shape, 
              "\n y_test shape: ", y_test.shape, separator)

    if args.method == 'all':
        classifiers.knn(x_train, y_train, x_test, y_test, args.verbose, unique_values)
        classifiers.dt(x_train, y_train, x_test, y_test, args.verbose, unique_values)
        classifiers.svc(x_train, y_train, x_test, y_test, args.verbose, unique_values)
        classifiers.mlp(x_train, y_train, x_test, y_test, args.verbose, unique_values)
        if args.verbose:
            print("finished all classifications", separator)
    else:    
        match args.method:
            case "knn":
                classifiers.knn(x_train, y_train, x_test, y_test, args.verbose, unique_values)
            case "dt":
                classifiers.dt(x_train, y_train, x_test, y_test, args.verbose, unique_values)
            case "svc":
                classifiers.svc(x_train, y_train, x_test, y_test, args.verbose, unique_values)
            case "mnb":
                classifiers.mlp(x_train, y_train, x_test, y_test, args.verbose, unique_values)
            case _:
                print("Invalid method", separator)



if __name__ == "__main__":
    main()