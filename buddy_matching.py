import csv
import copy
import math
import statistics
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Set
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

"""

v2.0: Bug fixes, buddy matching for new generic form, evaluation metrics
Ethan Pineda
Northwestern University
November 1, 2024


v1.0: Initial version
Alayna Richmond
Northwestern University
September 30, 2020
"""


# Column number
NAME = 2
PRONOUNS = 3
EMAIL = 4
YEAR = 5
SCHOOL = 6
CS_STATUS = 7
TIMEZONE = 8
ROLE = 9
TIME_COMMITMENT = 10

# Buddies
WAS_BUDDY_TRANSFER = 11
BUDDY_MATCH_ON_GENDER = 12
BUDDY_GENDER = 13
BUDDY_MATCH_ON_RACE = 14
BUDDY_RACE = 15


# New students
IS_NEW_STUDENT_TRANSFER = 16
NEW_STUDENT_CS_EXPERIENCE = 17
NEW_STUDENT_MATCH_ON_GENDER = 18
NEW_STUDENT_GENDER = 19
NEW_STUDENT_MATCH_ON_RACE = 20
NEW_STUDENT_RACE = 21


YEAR_VALUES = { "Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4, "Masters": 4 }


# CS Degree Statuses
DECLARED_MAJOR = 'Declared CS major'
DECLARED_MINOR = 'Declared CS minor'
INTENDED_MAJOR = 'Intend to declare CS major'
INTENDED_MINOR = 'Intend to declare CS minor'
UNDECLARED = 'Undeclared'


def parse_student_data(row):
    """Parse a single row of CSV data into student attributes."""
    try:
        name = row[2]  # Full name
        email = row[1]  # Email
        year = row[3]  # Graduation year
        school = row[4]  # School (McCormick, Weinberg, etc)
        cs_status = row[5]  # CS major/minor status
        pronouns = ""  # Not in current CSV
        
        # Time commitment
        try:
            time_commitment = int(row[8]) if row[8] else 1
        except (ValueError, IndexError):
            time_commitment = 1

        # Role determines parse path
        role = row[6]  # Mentor/Mentee role
        timezone = 0  # Default timezone

        if "Mentor" in role:
            transfer = "transfer student" in row[11].lower() if row[11] else False
            match_on_gender = "Yes" in str(row[12]) if row[12] else False
            gender = row[9] if row[9] else ""
            match_on_race = "Yes" in str(row[13]) if row[13] else False
            races = set(row[10].split(",")) if row[10] else set()
            cs_experience = 0  # Not applicable for mentors

        else:  # Mentee
            transfer = "transfer student" in row[11].lower() if row[11] else False
            match_on_gender = "Yes" in str(row[12]) if row[12] else False
            gender = row[9] if row[9] else ""
            match_on_race = "Yes" in str(row[13]) if row[13] else False
            races = set(row[10].split(",")) if row[10] else set()
            try:
                cs_experience = int(row[7]) if row[7] else 1
            except (ValueError, IndexError):
                cs_experience = 1

        return {
            'name': name,
            'email': email,
            'pronouns': pronouns,
            'year': year,
            'school': school,
            'cs_status': cs_status,
            'transfer': transfer,
            'timezone': timezone,
            'time_commitment': time_commitment,
            'match_on_gender': match_on_gender,
            'gender': gender,
            'match_on_race': match_on_race,
            'races': races,
            'cs_experience': cs_experience,
            'role': role
        }
    except Exception as e:
        print(f"Error parsing row: {e}")
        print(f"Row data: {row}")
        return None



def generate_mentors_and_mentees_from_survey_responses(filename):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header row

        mentors = []
        mentees = []
        skipped_2023 = 0
        total_rows = 0

        for row in reader:
            total_rows += 1
            timestamp = row[0]
            
            # Skip entries from 2023
            if '/2023' in timestamp:
                skipped_2023 += 1
                continue
                
            data = parse_student_data(row)
            if not data:
                continue

            if "Mentor" in data['role']:
                mentor = Student(
                    name=data['name'],
                    email=data['email'],
                    pronouns=data['pronouns'],
                    year=data['year'],
                    school=data['school'],
                    cs_status=data['cs_status'],
                    transfer=data['transfer'],
                    timezone=data['timezone'],
                    time_commitment=data['time_commitment'],
                    match_on_gender=data['match_on_gender'],
                    gender=data['gender'],
                    match_on_race=data['match_on_race'],
                    races=data['races']
                )
                mentors.append(mentor)
            else:
                mentee = NewStudent(
                    name=data['name'],
                    email=data['email'],
                    pronouns=data['pronouns'],
                    year=data['year'],
                    school=data['school'],
                    cs_status=data['cs_status'],
                    transfer=data['transfer'],
                    timezone=data['timezone'],
                    time_commitment=data['time_commitment'],
                    match_on_gender=data['match_on_gender'],
                    gender=data['gender'],
                    match_on_race=data['match_on_race'],
                    races=data['races'],
                    cs_experience=data['cs_experience']
                )
                mentees.append(mentee)

        print(f"\nParsing Summary:")
        print(f"Total rows processed: {total_rows}")
        print(f"Entries from 2023 skipped: {skipped_2023}")
        print(f"Successfully parsed {len(mentors)} mentors and {len(mentees)} mentees from 2024")
        
        if not mentors:
            print("Warning: No mentors found from 2024!")
        if not mentees:
            print("Warning: No mentees found from 2024!")
            
        return np.array(mentees), mentors
@dataclass
class Student:

    name: str
    pronouns: str
    email: str

    year: str
    school: str
    cs_status: str
    transfer: bool

    timezone: float
    time_commitment: int

    match_on_gender: bool
    gender: str
    match_on_race: bool
    races: Set[str]

@dataclass
class NewStudent(Student):

    cs_experience: int

    def similarity_to_student(self, student):
        pass

    def similarity_to_buddy(self, other_student):
        similarity = 0

        # Prefer to match students who are in the same school
        if self.school == other_student.school:
            similarity += 1
        
        # Prefer to match students pursuing same CS degree
        if 'major' in self.cs_status and 'major' in other_student.cs_status:
            similarity += 0.5
        if 'minor' in self.cs_status and 'minor' in other_student.cs_status:
            similarity += 0.5

        # Prefer to match transfers with each other
        if self.transfer and other_student.transfer:
            similarity += 6

        # Lower similarity for differences in time commitment
        similarity -= abs(self.time_commitment - other_student.time_commitment)

        # If both students want gender match and share gender identity
        if self.match_on_gender and other_student.match_on_gender and self.gender == other_student.gender:
            similarity += 6
        
        # Race matching
        if self.match_on_race and other_student.match_on_race:
            num_races_overlap = len(self.races & other_student.races)
            num_races = len(self.races)
            similarity += 4 * num_races_overlap / num_races

        return similarity

    
    def similarity(self, other_student):
        similarity = 0

        # School matching
        if self.school == other_student.school:
            similarity += 1
        
        # CS degree matching
        if 'major' in self.cs_status and 'major' in other_student.cs_status:
            similarity += 1
        if 'minor' in self.cs_status and 'minor' in other_student.cs_status:
            similarity += 1

        # Transfer student matching
        if self.transfer and other_student.transfer:
            similarity += 2

        # Time commitment difference penalty
        similarity -= abs(self.time_commitment - other_student.time_commitment)

        # Gender matching
        if self.match_on_gender and other_student.match_on_gender and self.gender == other_student.gender:
            similarity += 2
        
        # Race matching
        if self.match_on_race and other_student.match_on_race:
            num_races_overlap = len(self.races & other_student.races)
            num_races = len(self.races)
            similarity += 2 * num_races_overlap / num_races

        # CS experience similarity
        similarity -= abs(self.cs_experience - other_student.cs_experience) * 0.5

        return similarity

def new_student_similarity_matrix(new_students):
    num_students = len(new_students)
    similarity_matrix = np.ma.zeros((num_students, num_students))

    for i, student1 in enumerate(new_students):
        for j, student2 in enumerate(new_students):
            if i == j:
                similarity_matrix[i, j] = np.ma.masked
            else:
                similarity_matrix[i, j] = student1.similarity(student2)
        
    return similarity_matrix

def group_new_students(new_students, group_size=3, max_groups=None):
    """
    Create groups of three students, including handling remainders.
    """
    similarity_matrix = new_student_similarity_matrix(new_students)
    groups = []
    students_list = list(new_students)
    matched_students = set()
    
    while len(matched_students) < len(students_list):
        # Get remaining students
        available_indices = [i for i in range(len(students_list)) 
                           if students_list[i].email not in matched_students]
        
        if not available_indices:
            break
            
        # For last group, adjust size if needed
        current_size = min(group_size, len(available_indices))
        
        # Find the most dissimilar student among remaining students
        available_matrix = similarity_matrix[available_indices][:, available_indices]
        relative_outlier_idx = find_farthest_outlier(
            [students_list[i] for i in available_indices], 
            available_matrix
        )
        outlier_idx = available_indices[relative_outlier_idx]
        outlier = students_list[outlier_idx]
        
        # Initialize new group with outlier
        current_group = [outlier]
        matched_students.add(outlier.email)
        available_indices.remove(outlier_idx)
        
        # Find the most similar remaining students for this group
        while len(current_group) < current_size and available_indices:
            best_similarity = float('-inf')
            best_idx = None
            best_student = None
            
            for idx in available_indices:
                student = students_list[idx]
                if student.email not in matched_students:
                    avg_similarity = sum(student.similarity(other) 
                                      for other in current_group) / len(current_group)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_idx = idx
                        best_student = student
            
            if best_student is None:
                break
                
            current_group.append(best_student)
            matched_students.add(best_student.email)
            available_indices.remove(best_idx)
        
        if current_group:  # Add any non-empty group
            groups.append(np.array(current_group))
            
    return groups

def assign_buddies_to_groups(groups, buddies, max_mentors_per_group=1):
    """
    Assign mentors to groups with priority for complete groups and highest similarity.
    
    Args:
        groups: List of mentee groups
        buddies: List of available mentors
        max_mentors_per_group: Maximum mentors per group (default 1)
    
    Returns:
        List of tuples (group, assigned_mentors)
    """
    similarity_matrix = similarity_matrix_buddies_to_groups(groups, buddies)
    assigned_buddies = set()  # Track which mentors have been assigned
    groups_to_buddies = [[] for _ in groups]
    
    # Sort groups by size and create index mapping
    group_indices = list(range(len(groups)))
    group_indices.sort(key=lambda i: len(groups[i]), reverse=True)
    
    # First pass - assign mentors to complete groups only
    for group_idx in group_indices:
        if len(groups[group_idx]) < 2:  # Skip incomplete groups
            continue
            
        if len(assigned_buddies) >= len(buddies):  # No more available mentors
            break
            
        # Find best available buddy for this group
        best_buddy = None
        best_similarity = float('-inf')
        
        for buddy in buddies:
            if buddy.email in assigned_buddies:
                continue
                
            # Calculate total similarity between buddy and all students in group
            similarity = sum(student.similarity_to_buddy(buddy) 
                           for student in groups[group_idx])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_buddy = buddy
        
        # Assign best buddy if found
        if best_buddy:
            groups_to_buddies[group_idx].append(best_buddy)
            assigned_buddies.add(best_buddy.email)
    
    # Second pass - if any mentors remain, assign to incomplete groups
    if len(assigned_buddies) < len(buddies):
        for group_idx in group_indices:
            if len(groups_to_buddies[group_idx]) >= max_mentors_per_group:
                continue
                
            if len(assigned_buddies) >= len(buddies):
                break
                
            # Find best remaining buddy for this group
            best_buddy = None
            best_similarity = float('-inf')
            
            for buddy in buddies:
                if buddy.email in assigned_buddies:
                    continue
                    
                similarity = sum(student.similarity_to_buddy(buddy) 
                               for student in groups[group_idx])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_buddy = buddy
            
            if best_buddy:
                groups_to_buddies[group_idx].append(best_buddy)
                assigned_buddies.add(best_buddy.email)
    
    # Return groups paired with their assigned mentors
    return [(group, group_buddies) for group, group_buddies 
            in zip(groups, groups_to_buddies)]

def similarity_matrix_buddies_to_groups(groups, buddies):
    num_groups = len(groups)
    num_buddies = len(buddies)
    similarity_matrix = np.ma.zeros((num_buddies, num_groups))

    # Compute the similarity between each buddy and every group of new students
    for i, buddy in enumerate(buddies):
        for j, group in enumerate(groups):
            if i == j:
                similarity_matrix[i, j] = np.ma.masked
            else:
                similarity = 0
                for new_student in group:
                    similarity += new_student.similarity_to_buddy(buddy)
                similarity_matrix[i, j] = similarity
    
    return similarity_matrix



def assign_buddies_to_groups(groups, buddies, max_mentors_per_group=1):
    """Assign mentors to groups, ensuring each mentor is used only once."""
    orig_similarity_matrix = similarity_matrix_buddies_to_groups(groups, buddies)
    seen_buddies = set()  # Track mentor emails instead of indices
    groups_to_buddies = [[] for _ in groups]
    
    # First pass - assign primary mentors
    for group_idx in range(len(groups)):
        if len(seen_buddies) >= len(buddies):
            break
            
        available_buddies = [b for b in buddies if b.email not in seen_buddies]
        if not available_buddies:
            break
            
        # Calculate similarities for available buddies
        best_buddy = None
        best_similarity = float('-inf')
        for buddy in available_buddies:
            similarity = sum(student.similarity_to_buddy(buddy) 
                           for student in groups[group_idx])
            if similarity > best_similarity:
                best_similarity = similarity
                best_buddy = buddy
        
        if best_buddy:
            groups_to_buddies[group_idx].append(best_buddy)
            seen_buddies.add(best_buddy.email)
    
    return [(group, group_buddies) for group, group_buddies 
            in zip(groups, groups_to_buddies)]


def find_closest_buddy(similarity_matrix, farthest_group_ind):
    return np.argmax(similarity_matrix[:, farthest_group_ind])

def find_farthest_group(similarity_matrix):

    # For every point, determine the similarity to the closest point
    closest_similarities = np.ma.amax(similarity_matrix, axis=0)
    # Out of all points, find the one that is farthest from its closest point
    index_of_outlier = np.ma.argmin(closest_similarities)
    return index_of_outlier

def find_farthest_outlier(new_students, similarity_matrix):
    """
    Finds the point that is farthest away from all other points.
    Returns the index of the point in the similarity matrix
    """

    # student_has_request = [student.match_on_gender or student.match_on_race for student in new_students]
    # students_with_requests_inds = np.argwhere(student_has_request)
    # similarity_matrix = similarity_matrix.copy()
    # print("Before ")
    # print(similarity_matrix)
    # similarity_matrix[students_with_requests_inds, :] -= 2
    # print("After ")
    # print(similarity_matrix)


    # For every point, determine the similarity to the closest point
    closest_similarities = np.ma.amax(similarity_matrix, axis=1)
    # Out of all points, find the one that is farthest from its closest point
    index_of_outlier = np.ma.argmin(closest_similarities)
    return index_of_outlier

def find_closest_n_students(similarity_matrix, student_index, n):
    similarities = similarity_matrix[student_index]
    # Finds indices of the maximum n similarities
    indicies_of_closest_n_students = np.ma.argsort(similarities, endwith=False)[-1 * n:]
    return indicies_of_closest_n_students

     



class Buddy(Student):
    def distanceToBuddy(buddy):
        pass

class Mentorship:
    def __init__(self, mentee, mentor):

        if mentor.email in MANUAL_MATCHES and mentee.email in MANUAL_MATCHES[mentor.email]:
            self.manual_value = 100
        elif mentor.email in MANUAL_NO_MATCHES and mentee.email in MANUAL_NO_MATCHES[mentor.email]:
            self.manual_value = -100
        else:
            self.manual_value = 0

        self.year_difference = YEAR_VALUES[mentor.year] - YEAR_VALUES[mentee.year]
        self.is_mentor_older = self.year_difference > 0

        num_interests_overlap = len(mentor.experienced_fields & mentee.interested_fields)
        num_mentee_interests = len(mentee.interested_fields)
        self.fraction_interests_overlap = num_interests_overlap / num_mentee_interests

        num_topics_overlap = len(mentor.knowledgeable_topics & mentee.desired_topics)
        num_mentee_topics = len(mentee.desired_topics)
        self.fraction_topics_overlap = num_topics_overlap / num_mentee_topics

        self.mentor_copy_number = mentor.copy_number

        if mentor.email == "alaynarichmond2021@u.northwestern.edu":
            w = self.weight() + 5*self.mentor_copy_number
            if w > 4:
                print(mentee.email, self.fraction_interests_overlap, self.fraction_topics_overlap, w)

    def weight(self):
        weight = 0
        weight += (1 if self.is_mentor_older else 0)
        weight += (1 if self.year_difference > 0 and self.year_difference < 3 else 0)
        weight += math.log( 10 * self.fraction_interests_overlap + 1)
        #weight += self.fraction_interests_overlap#math.log( 10 * self.fraction_interests_overlap + 1) #4 * self.fraction_topics_overlap #2 * math.log(self.fraction_interests_overlap + 1)
        weight += 0.6 * self.fraction_topics_overlap
        weight -= 5 * self.mentor_copy_number
        weight += self.manual_value
        # print(weight)
        return weight

    def print(self):
        print("Year diff: ", self.year_difference)
        print("Interests: ", self.fraction_interests_overlap)
        print("Topics: ", self.fraction_topics_overlap)
        print("Mentor copy: ", self.mentor_copy_number)

def analyze_group_statistics(mentees, mentors, groups_with_mentors):
    """Create a visualization of group matching statistics."""
    # Count students in complete groups (2 mentees + mentor)
    students_in_complete_groups = 0
    students_in_incomplete_groups = 0
    students_unmatched = len(mentees)  # Start with all unmatched

    for group, group_mentors in groups_with_mentors:
        if len(group) == 2 and group_mentors:  # Complete group
            students_in_complete_groups += len(group)
            students_unmatched -= len(group)
        elif len(group) > 0:  # Incomplete group
            students_in_incomplete_groups += len(group)
            students_unmatched -= len(group)

    # Create pie chart of student distribution
    plt.figure(figsize=(10, 8))
    sizes = [students_in_complete_groups, students_in_incomplete_groups, students_unmatched]
    labels = ['Complete Groups', 'Incomplete Groups', 'Unmatched']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Student Matching Status', pad=20)
    plt.axis('equal')
    plt.savefig('student_distribution.png')
    plt.close()

    # Print detailed statistics
    print("\nStudent Matching Statistics:")
    print(f"Total students: {len(mentees)}")
    print(f"Students in complete groups: {students_in_complete_groups} ({students_in_complete_groups/len(mentees):.1%})")
    print(f"Students in incomplete groups: {students_in_incomplete_groups} ({students_in_incomplete_groups/len(mentees):.1%})")
    print(f"Unmatched students: {students_unmatched} ({students_unmatched/len(mentees):.1%})")

    # Analyze group sizes
    group_sizes = {}
    for group, mentors in groups_with_mentors:
        size = len(group)
        group_sizes[size] = group_sizes.get(size, 0) + 1

    print("\nGroup Size Distribution:")
    for size, count in sorted(group_sizes.items()):
        print(f"Groups with {size} students: {count}")

    print("\nMentor Distribution:")
    mentor_counts = {}
    for _, group_mentors in groups_with_mentors:
        mentor_count = len(group_mentors)
        mentor_counts[mentor_count] = mentor_counts.get(mentor_count, 0) + 1

    for count, groups in sorted(mentor_counts.items()):
        print(f"Groups with {count} mentor{'s' if count != 1 else ''}: {groups}")

    # Create group composition visualization
    plt.figure(figsize=(12, 6))
    x = range(len(group_sizes))
    plt.bar(x, group_sizes.values())
    plt.xticks(x, [f"{size} Students" for size in group_sizes.keys()])
    plt.title('Group Size Distribution')
    plt.ylabel('Number of Groups')
    plt.xlabel('Group Size')

    # Add value labels on bars
    for i, v in enumerate(group_sizes.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')

    plt.savefig('group_distribution.png')
    plt.close()

    return group_sizes, mentor_counts


def analyze_matches(mentees, mentors, groups_with_mentors):
    """Analyze matches and create visualizations of the matching statistics."""
    # Track matched and unmatched people
    matched_mentees = set()
    matched_mentors = set()
    mentor_load = defaultdict(int)
    
    # Collect matching statistics
    school_matches = []
    gender_matches = []
    race_matches = []
    cs_status_matches = []
    time_commitment_diffs = []
    cs_experience_diffs = []
    
    for group, group_mentors in groups_with_mentors:
        # Track matches
        for student in group:
            matched_mentees.add(student.email)
        for mentor in group_mentors:
            matched_mentors.add(mentor.email)
            mentor_load[mentor.email] += 1
            
        # Collect statistics
        if len(group) == 2:
            student1, student2 = group[0], group[1]
            school_matches.append(student1.school == student2.school)
            gender_matches.append(student1.gender == student2.gender)
            race_matches.append(bool(student1.races & student2.races))
            cs_status_matches.append(student1.cs_status == student2.cs_status)
            time_commitment_diffs.append(abs(student1.time_commitment - student2.time_commitment))
            cs_experience_diffs.append(abs(student1.cs_experience - student2.cs_experience))
    
    # Find unmatched people
    unmatched_mentees = {mentee.email: mentee for mentee in mentees if mentee.email not in matched_mentees}
    unmatched_mentors = {mentor.email: mentor for mentor in mentors if mentor.email not in matched_mentors}

    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Matching Success Rates
    plt.subplot(2, 2, 1)
    success_rates = {
        'School': sum(school_matches) / len(school_matches),
        'Gender': sum(gender_matches) / len(gender_matches),
        'Race': sum(race_matches) / len(race_matches),
        'CS Status': sum(cs_status_matches) / len(cs_status_matches)
    }
    plt.bar(success_rates.keys(), success_rates.values())
    plt.title('Matching Success Rates')
    plt.ylabel('Percentage')
    for i, v in enumerate(success_rates.values()):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center')

    # 2. Distribution of Time Commitment Differences
    plt.subplot(2, 2, 2)
    sns.histplot(time_commitment_diffs, bins=5)
    plt.title('Time Commitment Differences')
    plt.xlabel('Difference in Time Commitment')
    
    # 3. Distribution of CS Experience Differences
    plt.subplot(2, 2, 3)
    sns.histplot(cs_experience_diffs, bins=5)
    plt.title('CS Experience Differences')
    plt.xlabel('Difference in CS Experience')
    
    # 4. Mentor Load Distribution
    plt.subplot(2, 2, 4)
    mentor_loads = list(mentor_load.values())
    sns.histplot(mentor_loads, bins=range(min(mentor_loads), max(mentor_loads) + 2, 1))
    plt.title('Mentor Load Distribution')
    plt.xlabel('Number of Mentees per Mentor')
    
    plt.tight_layout()
    plt.savefig('matching_statistics.png')
    plt.close()

    # Print unmatched people
    print("\nUnmatched Mentees:")
    if unmatched_mentees:
        for email, mentee in unmatched_mentees.items():
            print(f"- {mentee.name} ({email})")
            print(f"  School: {mentee.school}")
            print(f"  CS Status: {mentee.cs_status}")
            print(f"  CS Experience: {mentee.cs_experience}")
    else:
        print("All mentees were matched!")

    print("\nUnmatched Mentors:")
    if unmatched_mentors:
        for email, mentor in unmatched_mentors.items():
            print(f"- {mentor.name} ({email})")
            print(f"  School: {mentor.school}")
            print(f"  CS Status: {mentor.cs_status}")
    else:
        print("All mentors were assigned mentees!")

    # Print mentor load statistics
    print("\nMentor Load Statistics:")
    load_counts = defaultdict(int)
    for load in mentor_loads:
        load_counts[load] += 1
    
    for load in sorted(load_counts.keys()):
        print(f"Mentors with {load} mentee{'s' if load != 1 else ''}: {load_counts[load]}")

    

    return unmatched_mentees, unmatched_mentors


class MentorshipGraph:
    def __init__(self, mentees, mentors):
        self.graph = nx.Graph()
        self.mentee_nodes = []
        self.mentor_nodes = []
        self.add_mentee_nodes_to_graph(mentees)
        self.add_mentor_nodes_to_graph(mentors)
        self.add_mentorship_edges_to_graph()

    def add_mentee_nodes_to_graph(self, mentees):
        self.mentee_nodes = mentees
        self.graph.add_nodes_from(mentees, bipartite=0)

    def add_mentor_nodes_to_graph(self, mentors):
        for mentor in mentors:
            num_mentees = mentor.num_mentees_possible()
            for _ in range(num_mentees):
                dup = mentor.duplicate()
                self.mentor_nodes.append(dup)
                self.graph.add_node(dup, bipartite=1)

    def add_mentorship_edges_to_graph(self):
        for mentee in self.mentee_nodes:
            for mentor in self.mentor_nodes:
                mentorship = Mentorship(mentee, mentor)
                self.graph.add_edge(mentee, mentor, mentorship=mentorship, weight=(-mentorship.weight()))

    def find_optimal_pairings(self):
        pairings = nx.bipartite.minimum_weight_full_matching(self.graph, self.mentee_nodes)
        return pairings

    def save_optimal_pairings_to_csv(self, pairings):
        with open('matches5.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            interest_fractions = []
            topic_fractions = []

            for key, value in pairings.items():
                if type(key) is Mentor:
                    mentor = key
                    mentee = value
                    mentorship = self.graph.edges[mentor, mentee]["mentorship"]
                    weight = self.graph.edges[mentor, mentee]["weight"]
                    interest_fractions.append(mentorship.fraction_interests_overlap)
                    topic_fractions.append(mentorship.fraction_topics_overlap)
                    writer.writerow([mentor.email, mentee.email, mentorship.is_mentor_older, mentorship.year_difference, mentorship.fraction_interests_overlap, mentorship.fraction_topics_overlap, weight, mentee.interested_fields & mentor.experienced_fields])
                    print(mentee.email)
                    print(mentor.email)
                    print(mentorship.is_mentor_older)
                    print("\n")

            print("\n")
            print("stddev (all): ", statistics.stdev(interest_fractions))
            print("avg (all): ", statistics.mean(interest_fractions))

            nonzero_interest_fractions = [x for x in interest_fractions if x != 0]
            print("stddev (nonzero): ", statistics.stdev(nonzero_interest_fractions))
            print("avg (nonzero): ", statistics.mean(nonzero_interest_fractions))


            print("\n")
            print("stddev (all): ", statistics.stdev(topic_fractions))
            print("avg (all): ", statistics.mean(topic_fractions))

def evaluate_new_student_groups(new_students, groups):
    percentage_school_matches = 0
    
    num_exact_cs_status_matches = 0
    num_close_cs_status_matches = 0
    
    timezone_diff_avg = 0
    time_commitment_diff_avg = 0

    num_transfers = 0
    num_transfers_matched = 0

    num_gender_requests_satisfied = 0
    num_gender_requests = 0
    percentage_gender_requests_satisfied = 0

    num_race_requests_satisfied = 0
    num_race_requests = 0
    percentage_race_requests_satisfied = 0

    cs_experience_diff_avg = 0

    num_students = len(new_students)
    num_groups = len(groups)

    for group in groups:
        student1 = group[0]
        student2 = group[1]

        print(student1.cs_experience, student2.cs_experience)

        if student1.school == student2.school:
            percentage_school_matches += 1 / num_groups

        if student1.cs_status == student2.cs_status:
            num_exact_cs_status_matches += 2
        elif 'major' in student1.cs_status and 'major' in student1.cs_status:
            num_close_cs_status_matches += 2
        elif 'minor' in student1.cs_status and 'minor' in student2.cs_status:
            num_close_cs_status_matches += 2

        if student1.transfer:
            num_transfers += 1
        if student2.transfer:
            num_transfers += 1
        if student1.transfer and student2.transfer:
            num_transfers += 2
        
        if student1.match_on_gender:
            num_gender_requests += 1
        if student2.match_on_gender:
            num_gender_requests += 1
        if student1.match_on_gender and student2.match_on_gender and student1.gender == student2.gender:
            num_gender_requests_satisfied += 2

        if student1.match_on_race:
            num_race_requests += 1
        if student2.match_on_race:
            num_race_requests += 1
        if student1.match_on_race and student2.match_on_race and len(student1.races & student2.races) > 0:
            num_race_requests_satisfied += 2

        
        timezone_diff_avg += abs(student1.timezone - student2.timezone) / num_groups
        time_commitment_diff_avg += abs(student1.time_commitment - student2.time_commitment) / num_groups
        cs_experience_diff_avg += abs(student1.cs_experience - student2.cs_experience) / num_groups


    print("Percentage school matches: ", percentage_school_matches)
    print("Percentage exact cs status matches :", num_exact_cs_status_matches / num_students)
    print("Percentage close cs status matches :", num_close_cs_status_matches / num_students)
    print("Percentage transfers matched: ", num_transfers_matched / num_transfers, " out of ", num_transfers)
    print("Timezone difference average: ", timezone_diff_avg)
    print("Time committment difference average: ", time_commitment_diff_avg)
    print("CS experience difference average: ", cs_experience_diff_avg)
    print("Percentage gender requests satisfied: ", num_gender_requests_satisfied / num_gender_requests, " out of ", num_gender_requests)
    print("Percentage race requests satisfied: ", num_race_requests_satisfied / num_race_requests, " out of ", num_race_requests)

def evaluate_matches(new_students, buddies, groups_with_buddies):
    if not new_students.size or not buddies or not groups_with_buddies:
        print("No matches to evaluate - empty data")
        return

    percentage_school_matches = 0
    num_exact_cs_status_matches = 0
    num_close_cs_status_matches = 0
    num_transfers = 0
    num_transfers_matched = 0
    num_gender_requests_satisfied = 0
    num_gender_requests = 0
    num_race_requests_satisfied = 0
    num_race_requests = 0

    # Student pair metrics
    cs_experience_diffs = []
    timezone_diffs = []
    time_commitment_diffs = []
    
    # Buddy-student metrics
    buddy_school_matches = 0
    total_buddy_student_pairs = 0
    buddy_gender_matches = 0
    buddy_race_matches = 0
    buddy_time_commitment_diffs = []
    buddy_timezone_diffs = []

    num_students = len(new_students)
    num_groups = len(groups_with_buddies)

    for group, group_buddies in groups_with_buddies:
        if len(group) < 2 or not group_buddies:  # Skip incomplete groups
            continue

        student1, student2 = group[0], group[1]

        # Print debug info
        print(f"Group info:")
        for student in group:
            print(f"Student: {student.time_commitment}, {student.timezone}, {student.school}, "
                  f"{student.cs_status}, {student.match_on_gender}, {student.gender}, "
                  f"{student.match_on_race}, {student.races}, {student.cs_experience}")
        for buddy in group_buddies:
            print(f"Buddy: {buddy.time_commitment}, {buddy.timezone}, {buddy.school}, "
                  f"{buddy.cs_status}, {buddy.match_on_gender}, {buddy.gender}, "
                  f"{buddy.match_on_race}, {buddy.races}")
        print()

        # Calculate student pair metrics
        if student1.school == student2.school:
            percentage_school_matches += 1
        
        cs_experience_diffs.append(abs(student1.cs_experience - student2.cs_experience))
        timezone_diffs.append(abs(student1.timezone - student2.timezone))
        time_commitment_diffs.append(abs(student1.time_commitment - student2.time_commitment))

        # Process student pair matching criteria
        for student in [student1, student2]:
            if student.transfer:
                num_transfers += 1
            if student.match_on_gender:
                num_gender_requests += 1
            if student.match_on_race:
                num_race_requests += 1

        if student1.cs_status == student2.cs_status:
            num_exact_cs_status_matches += 2
        elif ('major' in student1.cs_status and 'major' in student2.cs_status) or \
             ('minor' in student1.cs_status and 'minor' in student2.cs_status):
            num_close_cs_status_matches += 2

        if student1.match_on_gender and student2.match_on_gender and student1.gender == student2.gender:
            num_gender_requests_satisfied += 2

        if student1.match_on_race and student2.match_on_race and student1.races & student2.races:
            num_race_requests_satisfied += 2

        # Calculate buddy metrics
        for student in group:
            for buddy in group_buddies:
                total_buddy_student_pairs += 1
                
                if student.school == buddy.school:
                    buddy_school_matches += 1
                    
                if student.match_on_gender and buddy.match_on_gender and student.gender == buddy.gender:
                    buddy_gender_matches += 1
                    
                if student.match_on_race and buddy.match_on_race and student.races & buddy.races:
                    buddy_race_matches += 1
                    
                buddy_time_commitment_diffs.append(abs(student.time_commitment - buddy.time_commitment))
                buddy_timezone_diffs.append(abs(student.timezone - buddy.timezone))

    # Calculate and print metrics
    print("\nStudent Pair Metrics:")
    print(f"Percentage school matches: {percentage_school_matches/num_groups:.2%}")
    print(f"Percentage exact CS status matches: {num_exact_cs_status_matches/num_students:.2%}")
    print(f"Percentage close CS status matches: {num_close_cs_status_matches/num_students:.2%}")
    
    if num_transfers > 0:
        print(f"Percentage transfers matched: {num_transfers_matched/num_transfers:.2%} ({num_transfers_matched}/{num_transfers})")
    
    if cs_experience_diffs:
        print(f"CS experience difference average: {sum(cs_experience_diffs)/len(cs_experience_diffs):.2f}")
    if timezone_diffs:
        print(f"Timezone difference average: {sum(timezone_diffs)/len(timezone_diffs):.2f}")
    if time_commitment_diffs:
        print(f"Time commitment difference average: {sum(time_commitment_diffs)/len(time_commitment_diffs):.2f}")
    
    if num_gender_requests > 0:
        print(f"Gender request satisfaction: {num_gender_requests_satisfied/num_gender_requests:.2%} ({num_gender_requests_satisfied}/{num_gender_requests})")
    if num_race_requests > 0:
        print(f"Race request satisfaction: {num_race_requests_satisfied/num_race_requests:.2%} ({num_race_requests_satisfied}/{num_race_requests})")

    print("\nBuddy-Student Metrics:")
    if total_buddy_student_pairs > 0:
        print(f"Percentage of buddy school matches: {buddy_school_matches/total_buddy_student_pairs:.2%}")
        print(f"Percentage of buddy gender matches: {buddy_gender_matches/total_buddy_student_pairs:.2%}")
        print(f"Percentage of buddy race matches: {buddy_race_matches/total_buddy_student_pairs:.2%}")
        
        if buddy_time_commitment_diffs:
            print(f"Buddy time commitment difference average: {sum(buddy_time_commitment_diffs)/len(buddy_time_commitment_diffs):.2f}")
        if buddy_timezone_diffs:
            print(f"Buddy timezone difference average: {sum(buddy_timezone_diffs)/len(buddy_timezone_diffs):.2f}")

def group_new_students(new_students, group_size=3, max_groups=None):
    """
    Create groups of three students with a limit on total groups.
    
    Args:
        new_students: Array of students to group
        group_size: Desired size of each group (now defaults to 3)
        max_groups: Maximum number of groups to create (e.g., number of available mentors)
    
    Returns:
        List of student groups
    """
    similarity_matrix = new_student_similarity_matrix(new_students)
    groups = []
    students_list = list(new_students)
    matched_students = set()  # Track which students have been matched using email
    
    # Calculate maximum possible groups
    max_possible_groups = len(students_list) // group_size
    if max_groups is not None:
        max_possible_groups = min(max_possible_groups, max_groups)
        
    while len(groups) < max_possible_groups and len(matched_students) < len(students_list):
        # Get remaining students
        available_indices = [i for i in range(len(students_list)) 
                           if students_list[i].email not in matched_students]
        
        if len(available_indices) < group_size:
            break
            
        # Find the most dissimilar student among remaining students
        available_matrix = similarity_matrix[available_indices][:, available_indices]
        relative_outlier_idx = find_farthest_outlier(
            [students_list[i] for i in available_indices], 
            available_matrix
        )
        outlier_idx = available_indices[relative_outlier_idx]
        outlier = students_list[outlier_idx]
        
        if outlier.email in matched_students:
            continue
            
        # Initialize new group with outlier
        current_group = [outlier]
        matched_students.add(outlier.email)
        available_indices.remove(outlier_idx)
        
        # Find the most similar remaining students for this group
        while len(current_group) < group_size and available_indices:
            best_similarity = float('-inf')
            best_idx = None
            best_student = None
            
            for idx in available_indices:
                student = students_list[idx]
                if student.email not in matched_students:
                    # Calculate average similarity with all current group members
                    avg_similarity = sum(student.similarity(other) 
                                      for other in current_group) / len(current_group)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_idx = idx
                        best_student = student
            
            if best_student is None:
                break
                
            current_group.append(best_student)
            matched_students.add(best_student.email)
            available_indices.remove(best_idx)
        
        # Add group if it's complete or we're at the end
        if len(current_group) == group_size or (len(groups) == max_possible_groups - 1):
            groups.append(np.array(current_group))
            
    return groups

def print_group_statistics(mentees, mentors, groups_with_mentors):
    """Print detailed statistics about group matches."""
    # Count students in various group types
    students_in_complete_groups = 0  # Groups with 2 mentees + mentor
    students_in_partial_groups = 0   # Groups with either 1 mentee or no mentor
    students_unmatched = len(mentees)  # Start with all unmatched
    
    # Count groups by composition
    groups_with_two_mentees_and_mentor = 0
    groups_with_two_mentees_no_mentor = 0
    groups_with_one_mentee_and_mentor = 0
    groups_with_one_mentee_no_mentor = 0

    for group, group_mentors in groups_with_mentors:
        group_size = len(group)
        has_mentor = len(group_mentors) > 0
        
        # Update student counts
        if group_size == 2 and has_mentor:
            students_in_complete_groups += 2
            students_unmatched -= 2
            groups_with_two_mentees_and_mentor += 1
        elif group_size == 2:
            students_in_partial_groups += 2
            students_unmatched -= 2
            groups_with_two_mentees_no_mentor += 1
        elif group_size == 1 and has_mentor:
            students_in_partial_groups += 1
            students_unmatched -= 1
            groups_with_one_mentee_and_mentor += 1
        elif group_size == 1:
            students_in_partial_groups += 1
            students_unmatched -= 1
            groups_with_one_mentee_no_mentor += 1

    # Print overall statistics
    print("\n=== Overall Matching Statistics ===")
    print(f"Total mentees: {len(mentees)}")
    print(f"Total mentors: {len(mentors)}")
    
    print("\n=== Student Distribution ===")
    print(f"Students in complete groups (2 mentees + mentor): {students_in_complete_groups} ({students_in_complete_groups/len(mentees):.1%})")
    print(f"Students in partial groups: {students_in_partial_groups} ({students_in_partial_groups/len(mentees):.1%})")
    print(f"Unmatched students: {students_unmatched} ({students_unmatched/len(mentees):.1%})")

    print("\n=== Group Distribution ===")
    print(f"Groups with 2 mentees + mentor: {groups_with_two_mentees_and_mentor}")
    print(f"Groups with 2 mentees, no mentor: {groups_with_two_mentees_no_mentor}")
    print(f"Groups with 1 mentee + mentor: {groups_with_one_mentee_and_mentor}")
    print(f"Groups with 1 mentee, no mentor: {groups_with_one_mentee_no_mentor}")

    # Analyze mentor distribution
    mentor_counts = {}
    for _, group_mentors in groups_with_mentors:
        mentor_count = len(group_mentors)
        mentor_counts[mentor_count] = mentor_counts.get(mentor_count, 0) + 1

    print("\n=== Mentor Assignment Distribution ===")
    for count, groups in sorted(mentor_counts.items()):
        print(f"Groups with {count} mentor{'s' if count != 1 else ''}: {groups}")


def verify_matches(csv_file):
    """Analyze CSV file for duplicate assignments and provide corrected statistics."""
    student_groups = {}  # email -> list of groups
    mentor_groups = {}   # email -> list of groups
    group_sizes = {}     # group number -> size
    groups_with_mentor = set()
    
    # Read and analyze CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        current_group = None
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if not row:  # Skip empty rows
                continue
                
            if row[0].startswith('Group'):
                current_group = int(row[0].split()[1])
                group_sizes[current_group] = 0
                continue
                
            if not row[0]:  # This is a member row
                role = row[1]
                name = row[2]
                email = row[3]
                
                if role == 'Mentee':
                    group_sizes[current_group] += 1
                    if email not in student_groups:
                        student_groups[email] = []
                    student_groups[email].append(current_group)
                    
                elif role == 'Mentor':
                    groups_with_mentor.add(current_group)
                    if email not in mentor_groups:
                        mentor_groups[email] = []
                    mentor_groups[email].append(current_group)

    # Find duplicates
    duplicate_students = {email: groups for email, groups in student_groups.items() if len(groups) > 1}
    duplicate_mentors = {email: groups for email, groups in mentor_groups.items() if len(groups) > 1}

    # Print analysis
    print("\n=== MATCH VERIFICATION REPORT ===")
    
    print("\nDuplicate Student Assignments:")
    if duplicate_students:
        for email, groups in duplicate_students.items():
            print(f"- {email} appears in groups: {groups}")
    else:
        print("No duplicate student assignments found")
        
    print("\nDuplicate Mentor Assignments:")
    if duplicate_mentors:
        for email, groups in duplicate_mentors.items():
            print(f"- {email} appears in groups: {groups}")
    else:
        print("No duplicate mentor assignments found")

    print("\nCorrected Statistics:")
    print(f"Total unique students: {len(student_groups)}")
    print(f"Total unique mentors: {len(mentor_groups)}")
    print(f"Total groups: {len(group_sizes)}")
    print(f"Groups with mentors: {len(groups_with_mentor)}")
    print(f"Groups without mentors: {len(group_sizes) - len(groups_with_mentor)}")

    print("\nGroup Size Distribution:")
    size_counts = {}
    for size in group_sizes.values():
        size_counts[size] = size_counts.get(size, 0) + 1
    for size, count in sorted(size_counts.items()):
        print(f"Groups with {size} students: {count}")

    print("\nMentor Coverage:")
    students_with_mentor = sum(2 for group in groups_with_mentor)  # Assuming 2 students per group
    total_students_in_groups = sum(group_sizes.values())
    print(f"Students in groups with mentor: {students_with_mentor} ({students_with_mentor/total_students_in_groups:.1%})")
    print(f"Students in groups without mentor: {total_students_in_groups - students_with_mentor} ({(total_students_in_groups - students_with_mentor)/total_students_in_groups:.1%})")

    # Group consistency check
    print("\nGroup Consistency Check:")
    irregular_groups = {group: size for group, size in group_sizes.items() if size != 2}
    if irregular_groups:
        print("Found groups with irregular sizes:")
        for group, size in irregular_groups.items():
            print(f"- Group {group}: {size} students")
    else:
        print("All groups have exactly 2 students")

    return duplicate_students, duplicate_mentors, student_groups, mentor_groups

def debug_student_counts(mentees, groups, unmatched_students):
    """Print detailed breakdown of student numbers"""
    print("\nStudent Count Analysis:")
    print(f"Total mentees in input data: {len(mentees)}")
    
    # Count students in groups
    students_in_groups = 0
    for group in groups:
        students_in_groups += len(group)
        print(f"Group size: {len(group)}")
    
    print(f"\nTotal students in groups: {students_in_groups}")
    print(f"Number of groups: {len(groups)}")
    print(f"Number of unmatched students: {len(unmatched_students)}")
    print(f"Total accounted for: {students_in_groups + len(unmatched_students)}")
    
    # Check for duplicates
    all_emails = set()
    duplicate_emails = set()
    for group in groups:
        for student in group:
            if student.email in all_emails:
                duplicate_emails.add(student.email)
            all_emails.add(student.email)
    
    if duplicate_emails:
        print("\nFound duplicate student assignments:")
        for email in duplicate_emails:
            print(f"- {email}")

def debug_student_counts(mentees, groups, unmatched_students):
    """Print detailed breakdown of student numbers and group compositions"""
    print("\nStudent Count Analysis:")
    print(f"Total mentees in input data: {len(mentees)}")
    
    # Print all students from input
    print("\nInput Students:")
    for student in mentees:
        print(f"- {student.name} ({student.email})")
    
    print("\nGroup Compositions:")
    # Count and show students in groups
    students_in_groups = set()
    for i, group in enumerate(groups):
        print(f"\nGroup {i+1} (size: {len(group)}):")
        for student in group:
            print(f"- {student.name} ({student.email})")
            students_in_groups.add(student.email)
    
    print(f"\nTotal unique students in groups: {len(students_in_groups)}")
    print(f"Number of groups: {len(groups)}")
    
    # Find students not in any group
    all_emails = {student.email for student in mentees}
    unassigned_emails = all_emails - students_in_groups
    
    print("\nStudents not in any group:")
    for email in unassigned_emails:
        student = next(s for s in mentees if s.email == email)
        print(f"- {student.name} ({student.email})")
    
    print(f"\nTotal students accounted for in groups: {len(students_in_groups)}")
    print(f"Total unassigned students: {len(unassigned_emails)}")
    print(f"Sum should equal total mentees ({len(mentees)}): {len(students_in_groups) + len(unassigned_emails)}")


if __name__ == "__main__":
    # Load mentees and mentors
    mentees_raw, mentors = generate_mentors_and_mentees_from_survey_responses("cs_mentorship_2024_responses.csv")

    all_emails = set()

    for mentee in mentees_raw:
        all_emails.add(mentee.email)

    for mentor in mentors:
        all_emails.add(mentor.email)    

    with open('unique_emails.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Email'])  # Header
        for email in sorted(all_emails):  # Sort alphabetically
            writer.writerow([email])

    print(f"\nEmail Summary:")
    print(f"Total unique emails: {len(all_emails)}")
    print("Saved to 'unique_emails.csv'")
    
    # Deduplicate mentees based on email
    unique_mentees = {}
    duplicates = set()
    for mentee in mentees_raw:
        if mentee.email in unique_mentees:
            duplicates.add(mentee.email)
        unique_mentees[mentee.email] = mentee
    
    mentees = list(unique_mentees.values())
    
    print(f"\nDeduplication Summary:")
    print(f"Original mentees: {len(mentees_raw)}")
    print(f"Unique mentees: {len(mentees)}")
    print("\nDuplicate entries found for:")
    for email in duplicates:
        print(f"- {unique_mentees[email].name} ({email})")
    
    print(f"\nLoaded {len(mentees)} unique mentees and {len(mentors)} mentors")
    
    # Create groups of 3 mentees
    groups = group_new_students(mentees, group_size=3)
    print(f"Created {len(groups)} mentee groups")
    
    # Track which students got matched
    matched_students = set()
    for group in groups:
        for student in group:
            matched_students.add(student.email)
    
    # Find unmatched students
    unmatched_students = [student for student in mentees if student.email not in matched_students]
    
    # Detailed debug output
    debug_student_counts(mentees, groups, unmatched_students)
    
    
    # Assign mentors to groups
    groups_with_mentors = assign_buddies_to_groups(groups, mentors, max_mentors_per_group=1)
    
    # Print unmatched student details
    print("\nUnmatched Students:")
    print(f"Total unmatched: {len(unmatched_students)}")
    for student in unmatched_students:
        print(f"- {student.name} ({student.email})")
        print(f"  School: {student.school}")
        print(f"  CS Status: {student.cs_status}")
        print(f"  Year: {student.year}")
        print(f"  CS Experience Level: {student.cs_experience}")
    
    # Save matches to CSV with clear marking of mentor status
    with open('mentor_matches.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Group', 'Role', 'Name', 'Email', 'Year', 'School'])
        
        # First write all matched groups
        for i, (group, group_mentors) in enumerate(groups_with_mentors):
            writer.writerow([f'Group {i+1}'])
            for student in group:
                writer.writerow(['', 'Mentee', student.name, student.email, student.year, student.school])
            if group_mentors:
                for mentor in group_mentors:
                    writer.writerow(['', 'Mentor', mentor.name, mentor.email, mentor.year, mentor.school])
            else:
                writer.writerow(['', 'Note', 'No mentor assigned - on waitlist'])
            writer.writerow([])
        
        # Then write unmatched students at the bottom
        if unmatched_students:
            writer.writerow(['Unmatched Students'])
            for student in unmatched_students:
                writer.writerow(['', 'Unmatched', student.name, student.email, student.year, student.school])
    
    # Print statistics
    complete_groups = sum(1 for g, m in groups_with_mentors if len(g) == 3 and m)
    students_in_complete_groups = complete_groups * 3
    total_students = len(mentees)
    
    print("\nMatching Statistics:")
    print(f"Total mentees: {total_students}")
    print(f"Students in complete groups (3 mentees + mentor): {students_in_complete_groups} ({students_in_complete_groups/total_students:.1%})")
    print(f"Number of mentors: {len(mentors)}")
    print(f"Number of complete groups: {complete_groups}")
    print(f"Number of unmatched students: {len(unmatched_students)} ({len(unmatched_students)/total_students:.1%})")



    #  graph = MentorshipGraph(mentees, mentors)
    #  pairings = graph.find_optimal_pairings()
    #  graph.save_optimal_pairings_to_csv(pairings)
     #
     # for u, v, weight in graph.graph.edges.data('weight'):
     #    if weight is not None:
     #        print(weight)

     #
     # mentee = mentees[0]
     # mentor = mentors[20]
     # mentorship = Mentorship(mentee, mentor)
     #
     # mentee.print()
     # mentor.print()
     # mentorship.print()
