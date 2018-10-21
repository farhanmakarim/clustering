import random

class KMedoids:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.labels_ = []
        self.medoids_ = []
        self.costs_ = []
        
    def fit(self, data):
        # init
        self.medoids_ = [-1] * len(data)
        self.costs_ = [-1] * len(data)
        medoids = random.sample(range(0, len(data)), self.n_clusters)
        # first assignment
        self.medoids_ , self.costs_ = self.assign_clusters(data, medoids)
        # swapping
        swapped_medoids = medoids.copy()
        remaining_objects =[x for x in range(0, len(data)) if x not in medoids]
        for i in range(0, len(medoids)):
            previous_cost = -1
            current_cost = sum(self.costs_)
            for j in range(0, len(remaining_objects)):
                # swap
                dummy = swapped_medoids[i]
                swapped_medoids[i] = remaining_objects[j]
                remaining_objects[j] = dummy
                
                # reassign
                new_medoids, new_costs = self.assign_clusters(data, swapped_medoids)
                calculated_cost = sum(new_costs)
                
                if (calculated_cost < previous_cost):
                    # update labels
                    self.medoids_ = new_medoids
                    self.costs_ = new_costs
                    previous_cost = calculated_cost
                else:
                    # swap back if the cost is higher
                    dummy = remaining_objects[j]
                    remaining_objects[j] = swapped_medoids[i]
                    swapped_medoids[i] = dummy
        # encode labels
        self.labels_ = [-1] * len(data)
        for i in range(0, len(data)):
            self.labels_[i] = swapped_medoids.index(self.medoids_[i])
        

    def assign_clusters(self, data, ro_pool):
        medoids = [-1] * len(data)
        costs = [1] * len(data)
        for i in range(0, len(data)):
            assigned_ro = -1
            cost = float('inf')
            for j in range(0, self.n_clusters):
                checked_object = data[i]
                checked_ro = data[ro_pool[j]]
                calculated_cost = self.manhattan_distance(checked_object, checked_ro)
                if (cost == -1) or (cost > calculated_cost):
                    cost = calculated_cost
                    assigned_ro = ro_pool[j]
            medoids[i] = assigned_ro
            costs[i] = cost
        return (medoids, costs)
            
    def manhattan_distance(self, p0, p1):
        distance = 0
        for i in range(0, len(p0)):
            distance += abs(p0[i] - p1[i])
        return distance