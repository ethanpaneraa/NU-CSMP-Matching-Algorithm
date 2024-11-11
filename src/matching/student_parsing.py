import csv
from dataclasses import dataclass
from typing import Set

YEAR_VALUES = { "Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4, "Masters": 4 }

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
            
        return mentees, mentors
