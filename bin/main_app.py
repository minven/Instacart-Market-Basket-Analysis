# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:55:54 2017

@author: minven2
"""

import pandas as pd # dataframes
import numpy as np # algebra & calculus
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
import pickle
import time 
import random
from scipy.sparse import csc_matrix
from skmultilearn.problem_transform import ClassifierChain
from sklearn import linear_model


def load_pickle(pickle_name):
    pickle_obj = pd.read_pickle( open( "../pickles/{}".format(pickle_name), "rb" ))       
    return pickle_obj  

def save_raw_data():
        """
        Initial raw data reading and saving data to pickles
        """
        aisles = pd.read_csv('../data/aisles.csv', engine='c')
        pickle.dump(aisles, open("../pickles/aisles.p", "wb"))
        
        products = pd.read_csv('../data/products.csv', engine='c')
        pickle.dump(products, open("../pickles/products.p", "wb"))
        
        departments = pd.read_csv('../data/departments.csv', engine='c')
        pickle.dump(departments, open("../pickles/departments.p", "wb"))
        
        op_train = pd.read_csv('../data/order_products__train.csv',
                               engine='c', 
                               dtype={'order_id': np.int32,
                                      'product_id': np.int32, 
                                      'add_to_cart_order': np.int16,
                                      'reordered': np.int8})
        pickle.dump(op_train, open("../pickles/op_train.p", "wb"))
    
        test = pd.read_csv('../data/sample_submission.csv', engine='c')
        pickle.dump(test, open("../pickles/test.p", "wb"))
        
        op_prior = pd.read_csv('../data/order_products__prior.csv', engine='c', 
                               dtype={'order_id': np.int32,
                                      'product_id': np.int32, 
                                      'add_to_cart_order': np.int16,
                                      'reordered': np.int8})
        pickle.dump(op_prior, open("../pickles/op_prior.p", "wb"))
        
        orders = pd.read_csv('../data/orders.csv', engine='c',
                             dtype={'order_id': np.int32, 
                                    'user_id': np.int32, 
                                    'order_number': np.int32, 
                                    'order_dow': np.int8, 
                                    'order_hour_of_day': np.int8, 
                                    'days_since_prior_order': np.float16})
        pickle.dump(orders, open("../pickles/orders.p", "wb"))
      




class Preprocessing(object):
    def __init__(self):
        self.data_sources = {"aisles": "aisles.p",
                              "products": "products.p",
                              "departments": "departments.p",
                              "op_train": "op_train.p",
                              "test": "test.p",
                              "op_prior": "op_prior.p",
                              "orders": "orders.p",
                              "goods": "goods.p",
                              "op_train_with_user": "op_train_with_user.p",
                              "op_prior_with_user": "op_prior_with_user.p",
                              "op_prior_and_train_merged": "op_prior_and_train_merged.p",
                              "products_grouped_by_users": "products_grouped_by_users.p",
                              "index_collection_for_each_user": "index_collection_for_each_user.p",
                              "users_products_dict": "users_products_dict.p",
                              "users_orders_count": "users_orders_count.p",
                              "users_products_count": "users_products_count.p",
                              "X_train_sparse": "X_train_sparse.p",
                              "y_train_sparse": "y_train_sparse.p",
                              "users_products_count_rdcd": "users_products_count_rdcd.p",
                              "X_train": "X_train.p",
                              "y_train": "y_train.p"}
    
    
    def save_goods(self):
        '''
        Join aisles, products, depeartments to one table of goods
        '''

        aisles = load_pickle(self.data_sources["aisles"])
        products = load_pickle(self.data_sources["products"])
        departments = load_pickle(self.data_sources["departments"])
        
        products_with_aisles = pd.merge(left=aisles,
                                        right=products,
                                        how="inner",
                                        on="aisle_id")
        
        goods = pd.merge(left=products_with_aisles,
                         right=departments,
                         how="inner",
                         on="department_id")
        # change cols order
        goods = goods[["product_id", "aisle_id", "department_id", "product_name" ,"aisle", "department"]]
        # change cols names
        goods.columns = ["product_id", "aisle_id", "department_id", "product" ,"aisle", "department"]
        pickle.dump(goods, open("../pickles/goods.p", "wb"))
        
    def save_op_train_with_user(self):
        """
        Join orders-products train with orders
        """
        op_train = load_pickle(self.data_sources["op_train"])
        orders = load_pickle(self.data_sources["orders"])
        op_train_with_user = pd.merge(left=op_train,
                                      right=orders,
                                      how="inner",
                                      on="order_id")
        pickle.dump(op_train_with_user, open("../pickles/op_train_with_user.p", "wb"))
        
    def save_op_prior_with_user(self):
        """
        Join orders-products prior with orders
        """
        op_prior = load_pickle(self.data_sources["op_prior"])
        orders = load_pickle(self.data_sources["orders"])
        op_prior_with_user = pd.merge(left=op_prior,
                                      right=orders,
                                      how="inner",
                                      on="order_id")
        pickle.dump(op_prior_with_user, open("../pickles/op_prior_with_user.p", "wb"))
        
    def save_op_prior_and_train_merged(self):
        """
        Merge train and Prior datasets
        """
        op_train_with_user = load_pickle(self.data_sources["op_train_with_user"])
        op_prior_with_user = load_pickle(self.data_sources["op_prior_with_user"])            
        op_prior_and_train_merged = op_prior_with_user.append(op_train_with_user)
        op_prior_and_train_merged["eval_set"] = op_prior_and_train_merged["eval_set"].map({"prior":"train"})
        pickle.dump(op_prior_and_train_merged, open("../pickles/op_prior_and_train_merged.p", "wb"))
        
        
    def save_products_grouped_by_users(self):
        """
        Products count for each user
        """
        op_prior_and_train_merged = load_pickle(self.data_sources["op_prior_and_train_merged"])
        goods = load_pickle(self.data_sources["goods"])
        products_grouped_by_users = op_prior_and_train_merged.groupby(["user_id", "product_id"]).size()
        products_grouped_by_users = pd.DataFrame(products_grouped_by_users)
        products_grouped_by_users.reset_index(inplace=True)
        products_grouped_by_users.columns = ["user_id", "product_id", "freq"]
         
        products_grouped_by_users = pd.merge(left=products_grouped_by_users,
                                                right=goods,
                                                how="inner",
                                                on="product_id")
        pickle.dump(products_grouped_by_users, open("../pickles/products_grouped_by_users.p", "wb"))
        
    def save_indexes_for_each_user(self):
        """
        collect indexes for each user for better querying performance
        """
        products_grouped_by_users = load_pickle(prepr.data_sources["products_grouped_by_users"])
        products_grouped_by_users = products_grouped_by_users[["user_id", "product_id", "freq"]]
        products_grouped_by_users.columns = ["user_id", "product_id", "count"]

        users_id = np.unique(products_grouped_by_users["user_id"])

        index_collection = {}
        count = 1
        for user_id in users_id:
            ith_user = products_grouped_by_users[products_grouped_by_users["user_id"] == user_id]
            index_collection[user_id] = ith_user.index 
            if count % 1000 == 0:
                print(count)
            count += 1
            
        pickle.dump(index_collection, open("../pickles/index_collection_for_each_user.p", "wb"))
        
    def save_products_for_each_user(self):
        """
        collect product ids for each user save them to dict
        """
        products_grouped_by_users = load_pickle(prepr.data_sources["products_grouped_by_users"])
        index_collection = load_pickle(prepr.data_sources["index_collection_for_each_user"])
        products_grouped_by_users = products_grouped_by_users[["user_id", "product_id", "freq"]]
        products_grouped_by_users.columns = ["user_id", "product_id", "count"]
        
        count = 1
        users_products_dict = {}
        for user_id in index_collection:
            users_products_dict[user_id] = products_grouped_by_users.iloc[index_collection[user_id]]["product_id"]
            if count % 1000 == 0:
                print(count)
            count += 1
        pickle.dump(users_products_dict, open("../pickles/users_products_dict.p", "wb"))
 

    def save_users_orders_eval_set(self):
        """
        Construct Training Data Set
        To do training I need to mark last known order as test set 
        and the rest ones as train set
        find last order id for each user    
        """
        
        op_prior_and_train_merged = load_pickle(prepr.data_sources["op_prior_and_train_merged"])

        # product count in each order also order_number included
        users_orders_count = op_prior_and_train_merged.groupby(["user_id", "order_number", "order_id"]).size()
        users_orders_count = users_orders_count.reset_index()
        users_orders_count.columns = ["user_id", "order_number", "order_id", "count"]
        del op_prior_and_train_merged

        del users_orders_count["count"]
        users_orders_count["eval_set"] = ""
        
        users_orders_count = users_orders_count.reset_index()
        del users_orders_count["index"]

        count = 1
        for i in users_orders_count.index[:-1]:
            if users_orders_count.iloc[i]["order_number"] < users_orders_count.iloc[i+1]["order_number"]:
                users_orders_count.set_value(i, "eval_set", "train")
            else:
                users_orders_count.set_value(i, "eval_set", "test")
            if count % 10000== 0:
                print(count)                
            count += 1
        pickle.dump(users_orders_count, open("../pickles/users_orders_count.p", "wb"))


    def save_users_products_eval_set(self):
        """
        """
        op_prior_and_train_merged = load_pickle("op_prior_and_train_merged.p")
        users_orders_count =  load_pickle("users_orders_count.p")
        
        users_products_count = op_prior_and_train_merged.groupby(["user_id",  "order_number", "order_id","product_id"]).size()
        users_products_count = users_products_count.reset_index()
        users_products_count.columns = ["user_id",  "order_number", "order_id", "product_id","count"]
        del users_products_count["count"]
        users_orders_count_tmp = op_prior_and_train_merged.groupby(["user_id", "product_id"]).size()
        users_orders_count_tmp = users_orders_count_tmp.reset_index()
        users_orders_count_tmp.columns = ["user_id", "product_id", "count"]
        del op_prior_and_train_merged
        
        users_products_count = pd.merge(left=users_products_count,
                                       right=users_orders_count[["user_id", "order_id","eval_set"]],
                                       how="inner",
                                       on=["user_id","order_id"])
        del users_orders_count
        
        users_products_count = pd.merge(left=users_products_count,
                                       right=users_orders_count_tmp,
                                       how="inner",
                                       on=["user_id","product_id"])
        
        pickle.dump(users_products_count, open("../pickles/users_products_count.p", "wb"))
    
    def plot_uniq_prod_for_each_dep(self):
        """
        unique products count for each department
        """
        goods = load_pickle(self.data_sources["goods"])
        
        plt.figure(figsize=(20, 12))
        goods.groupby(['department']).count()['product_id']\
        .sort_values(ascending=False).plot(kind='bar', 
                                           title='Unique products in each department')  
        plt.savefig("../visualizations/unique_products_in_each_department.pdf")

    def plot_uniq_prod_for_each_ai_in_dep(self):
        """
        unique products count for each aisle in each department
        """
        goods = load_pickle(self.data_sources["goods"])
        departments = load_pickle(self.data_sources["departments"])

        f, ax = plt.subplots(6, 4, figsize=(12, 30))
        grouping_series = goods.groupby(['department', "aisle"]).size()
        for i, department in enumerate(departments["department"]):
            #extract subset by department
            subset_series = grouping_series[grouping_series.index.get_level_values("department") == department]\
                                            .sort_values(ascending=False)
            aisles_names = list(subset_series.index.get_level_values("aisle"))
            aisles_counts = list(subset_series.values)
            y_pos = np.arange(len(aisles_counts))
            ax[i//4, i%4].bar(y_pos, aisles_counts, align='center', alpha=0.5)
            ax[i//4, i%4].set_xticks(y_pos)
            ax[i//4, i%4].set_xticklabels(aisles_names, rotation="vertical")
            ax[i//4, i%4].set_title('Dep: {}'. format(department))
        f.subplots_adjust(hspace=2)
        plt.savefig("../visualizations/unique_products_for_aisles.pdf")
        
    def plot_products_count_per_oder(self):
        """
        products count per oder
        """
        op_prior_and_train_merged = load_pickle(self.data_sources["op_prior_and_train_merged"])

        products_count_in_orders = op_prior_and_train_merged.groupby(["order_id"]).size()
        plt.figure(figsize=(10, 5))
        plt.hist(products_count_in_orders, bins=range(0, 50, 2))
        plt.title("products count in order")
        plt.savefig("../visualizations/products_count_in_orders_zoomed.pdf")
         
        plt.figure(figsize=(10, 5))
        plt.hist(products_count_in_orders, bins=range(0, max(products_count_in_orders), 2))
        plt.title("products count in order")
        plt.savefig("../visualizations/products_count_in_orders_all.pdf")
        
        
        
    def calculate_users_similarity(self):
        """
        Calculate similar users by identifying how similar baskets were
        """
        products_grouped_by_users = load_pickle(prepr.data_sources["products_grouped_by_users"])
        users_products_dict = load_pickle(prepr.data_sources["users_products_dict"])

        products_grouped_by_users = products_grouped_by_users[["user_id", "product_id", "freq"]]
        products_grouped_by_users.columns = ["user_id", "product_id", "count"]
        users_id = np.unique(products_grouped_by_users["user_id"])
        del products_grouped_by_users
        
        similiarity_by_users = {}
        count = 1
        for user_id in users_id:
            if count % 100 == 0:
                print(count)
            if (count > 200000) & (count<230001):
                #ith_user = users_product_size.iloc[index_collection[user_id]]
                ith_user = users_products_dict[user_id]
                ith_user_set = set(ith_user)
                # generate random user indexes to speed up process
                random_users_id = random.sample(range(1,206209), 5000)
                # Find n most similar users by products buying behavior
                similarity_by_products = {}
                for random_user in random_users_id:
                    #jth_user = users_product_size.iloc[index_collection[random_user]]
                    jth_user = users_products_dict[random_user]
                    itersec = list(ith_user_set.intersection(jth_user))
                    similarity_by_products[random_user] = len(itersec)
                
                most_similar = {}
                # collect most similar users and product counts
                for i in range(10):
                    # user wich is most similart to ith user. Max value
                    max_user = max(similarity_by_products, key=lambda i: similarity_by_products[i])
                    most_similar[max_user] = similarity_by_products[max_user]
                    similarity_by_products[max_user] = -1
                similiarity_by_users[user_id] = most_similar
            count += 1
        pickle.dump(similiarity_by_users, open("../pickles/similiarity_by_users_200000_206200.p", "wb"))
                
    def load_reduced_users_products_count(self):
        """
        remove products which were bought rarely between all users orders history
        """
        users_products_count = load_pickle(prepr.data_sources["users_products_count"])
        users_products_count_tmp = users_products_count.groupby(["product_id"]).size()
        users_products_count_tmp = users_products_count_tmp.reset_index()            
        users_products_count_tmp.columns = ["product_id", "count"]
        product_ids_del = users_products_count_tmp[users_products_count_tmp["count"] > 40]["product_id"]
        self.users_products_count_rdcd = users_products_count.loc[users_products_count['product_id'].isin(product_ids_del)]
        del users_products_count,users_products_count_tmp
        pickle.dump(self.users_products_count_rdcd, open("../pickles/users_products_count_rdcd.p", "wb"))  
        
    def users_for_train_and_test(self, train_sample = 50000):
        """
        Generate Train and Test samples from users
        """
        if not hasattr(self, 'users_products_count_rdcd'):
            self.users_products_count_rdcd = load_pickle(prepr.data_sources["users_products_count_rdcd"])
        random.seed(10)
        users_id = np.unique(self.users_products_count_rdcd["user_id"])    
        users_for_train = random.sample(list(users_id), k=train_sample)
        users_for_test = set(users_id)
        users_for_test.difference_update(set(users_for_train))
        users_for_test = list(users_for_test)
        
        self.users_for_train = users_for_train
        self.users_for_test = users_for_test
        

    def transform_to_sparse(self, subset, eval_set):
        subset_users =  np.unique(subset["user_id"])
        subset_products = np.unique(subset["product_id"])
        

        subset_users_mapping = dict(zip(subset_users, range(len(subset_users))))
        subset_products_mapping = dict(zip(subset_products, range(len(subset_products))))
        
        
        subset_users_mapped = subset["user_id"].map(subset_users_mapping) 
        subset_products_mapped = subset["product_id"].map(subset_products_mapping)
        if eval_set == "train":
            subset_count = subset["count"]
            pickle_name = "X_train_sparse.p"
        elif eval_set == "test":
            subset_count = pd.Series([1]*len(subset["count"]))
            pickle_name = "y_train_sparse.p"
            
        subset_sparse = csc_matrix((subset_count,(subset_users_mapped,subset_products_mapped)),
                                           shape=(len(subset_users_mapping), len(subset_products_mapping)))
        pickle.dump(subset_sparse, open("../pickles/{}".format(pickle_name), "wb"))

        

    def generate_X_train(self):
        """
        Generate X_train sparse matrix [users_count x products_count(all history except last one order)]
        values are counts for products bought by users 
        """
        if not hasattr(self, 'users_products_count_rdcd'):
            self.users_products_count_rdcd = load_pickle(prepr.data_sources["users_products_count_rdcd"])
        X_train = self.users_products_count_rdcd.loc[(self.users_products_count_rdcd['eval_set'] == 'train') & self.users_products_count_rdcd['user_id'].isin(self.users_for_train)]
        pickle.dump(X_train, open("../pickles/X_train.p", "wb"))
        

        
        
    def generate_y_train(self):
        """
        Generate y_train sparse matrix [users_count x products_count(last one order)]
        values ones for products which were bought at last order
        """
        if not hasattr(self, 'users_products_count_rdcd'):
            self.users_products_count_rdcd = load_pickle(prepr.data_sources["users_products_count_rdcd"])
        y_train = self.users_products_count_rdcd.loc[(self.users_products_count_rdcd['eval_set'] == 'test') & self.users_products_count_rdcd['user_id'].isin(self.users_for_train)]
        pickle.dump(y_train, open("../pickles/y_train.p", "wb"))
        
                
    
                  
    
if __name__ == "__main__":
    # read_raw_data()
    prepr = Preprocessing()
    
    # prepr.save_goods()
    # prepr.save_op_train_with_user()
    # op_train_with_user = load_pickle(prepr.data_sources["op_train_with_user"])
    # prepr.save_op_prior_with_user()
    # prepr.plot_uniq_prod_for_each_dep()
    # prepr.plot_uniq_prod_for_each_ai_in_dep()
    # prepr.save_op_prior_and_train_merged()
    # prepr.save_products_grouped_by_users()
    # products_grouped_by_users = load_pickle(prepr.data_sources["products_grouped_by_users"])
    # prepr.plot_products_count_per_oder()
    # users_products_dict = load_pickle(prepr.data_sources["users_products_dict"])
    # prepr.save_users_products_eval_set()
    
    # prepr.load_reduced_users_products_count()
    # prepr.users_for_train_and_test(train_sample=50000)
    # prepr.generate_X_train()
    # prepr.generate_y_train()
    X_train = load_pickle(prepr.data_sources["X_train"])
    y_train = load_pickle(prepr.data_sources["y_train"])
    # prepr.transform_to_sparse(X_train, "train")
    # prepr.transform_to_sparse(y_train, "test")
        
 
    
    """
    After unpopular products removal check if some users are without any
    products
    """    
    X_train_users =  np.unique(X_train["user_id"])
    y_train_users =  np.unique(y_train["user_id"])
    
    X_train_users_set = set(X_train_users)
    X_train_users_set.difference_update(set(y_train_users))
    users_difference = list(X_train_users_set)

    X_train = X_train.loc[~X_train['user_id'].isin(X_train_users_set)]
    y_train = y_train.loc[~y_train['user_id'].isin(X_train_users_set)]
    prepr.transform_to_sparse(X_train, "train")
    prepr.transform_to_sparse(y_train, "test")    
    X_train_sparse = load_pickle(prepr.data_sources["X_train_sparse"])
    y_train_sparse = load_pickle(prepr.data_sources["y_train_sparse"])    
    
    

        
    if False:
        users_products_dict = load_pickle(prepr.data_sources["users_products_dict"])
        users1 = load_pickle("similiarity_by_users_1_10000.p")
        users2 = load_pickle("similiarity_by_users_10000_100000.p")
        users3 = load_pickle("similiarity_by_users_100000_200000.p")
        users4 = load_pickle("similiarity_by_users_200000_206200.p")
        
        users_similarity = {}
        users_similarity.update(users1)
        users_similarity.update(users2)
        users_similarity.update(users3)
        users_similarity.update(users4)
        """
        Check how valid are users_similarity
        """
        count = 1
        for user_id_i, similarities_dict in users_similarity.items():
            user_id_i_count = len(users_products_dict[user_id_i])
            print("User {} bought {} products".format(user_id_i, user_id_i_count))
            for user_id_j, common_elements in similarities_dict.items():
                user_id_j_count = len(users_products_dict[user_id_j])
                if common_elements / user_id_j_count> 0.2:
                    print("User {} bought {} products and {} are the same".format(user_id_j, user_id_j_count,common_elements))
            if count % 100 == 0:
                break
            count += 1    

       
        
    if False:

        
        # Modeling Classifier Chains for Multi-label Classification


#        sums_across_labels = y_train_sparse.sum(axis=1)
#        noise = sums_across_labels > 2
#        noise = noise.transpose()
#        noise = noise[:].tolist()[0]
#        sums_across_labels[sums_across_labels == 1].shape
        
        
        

        X_train_sparse_csr = X_train_sparse.tocsr()
        y_train_sparse_csr = y_train_sparse.tocsr()

        classifier = ClassifierChain(classifier=linear_model.SGDClassifier(penalty="l1",n_jobs=-1),
                                     require_dense = [False, True])
        classifier.fit(X_train_sparse_csr, y_train_sparse_csr)            
        
 

    
    
        
        
        
        

        
        