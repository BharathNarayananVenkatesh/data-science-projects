CREATE TABLE country_vaccinations_total (
iso_code TEXT NOT NULL,
vaccine_name TEXT NOT NULL,
date TEXT,
source_url TEXT,
people_vaccinated TEXT,
people_fully_vaccinated TEXT,
total_boosters TEXT,
PRIMARY KEY (
iso_code,
vaccine_name,
date
),
FOREIGN KEY (
iso_code,
vaccine_name
)
REFERENCES vaccine (iso_code,
vaccine_name) 
);


CREATE TABLE locations (
iso_code TEXT PRIMARY KEY
NOT NULL,
location_name TEXT
);

CREATE TABLE us_state (
iso_code TEXT NOT NULL,
state_name TEXT,
PRIMARY KEY (
state_name,
iso_code
),
FOREIGN KEY (
iso_code
)
REFERENCES locations (iso_code) 
);


CREATE TABLE us_state_vaccinations (
iso_code VARCHAR NOT NULL,
state_name VARCHAR NOT NULL,
date DATE NOT NULL,
total_distributed INT,
people_vaccinated INT,
people_fully_vaccinated_per_hundred INT,
people_fully_vaccinated INT,
people_vaccinated_per_hundred INT,
distributed_per_hundred INT,
daily_vaccinations_raw INT,
daily_vaccinations INT,
daily_vaccinations_per_million INT,
share_doses_used INT,
total_boosters INT,
total_boosters_per_hundred INT,
PRIMARY KEY (
iso_code,
state_name,
date
),
FOREIGN KEY (
iso_code,
state_name
)
REFERENCES us_state (iso_code,
state_name) 
);

CREATE TABLE vaccination (
date DATE NOT NULL,
iso_code TEXT NOT NULL,
people_vaccinated INT,
people_fully_vaccinated INT,
total_boosters INT,
daily_vaccinations_raw INT,
daily_vaccinations INT,
people_vaccinated_per_hundred INT,
people_fully_vaccinated_per_hundred INT,
total_boosters_per_hundred INT,
daily_vaccinations_per_million INT,
daily_people_vaccinated INT,
daily_people_vaccinated_per_hundred INT,
totalVaccinations INT,
totalVaccinationsPerHundred INT,
PRIMARY KEY (
iso_code,
date
),
FOREIGN KEY (
iso_code
)
REFERENCES locations (iso_code) 
);

CREATE TABLE vaccinations_by_age_group (
iso_code TEXT NOT NULL,
age_group TEXT,
date DATE,
people_vaccinated_per_hundred REAL,
people_fully_vaccinated_per_hundred REAL,
people_with_booster_per_hundred REAL,
PRIMARY KEY (
iso_code,
age_group,
date
),
FOREIGN KEY (
iso_code
)
REFERENCES locations (iso_code) 
);


CREATE TABLE vaccinations_by_manufacturer (
iso_code TEXT NOT NULL,
vaccine_name TEXT NOT NULL,
date TEXT,
total_vaccinations INTEGER,
PRIMARY KEY (
iso_code,
vaccine_name,
date
),
FOREIGN KEY (
iso_code,
vaccine_name
)
REFERENCES vaccine (iso_code,
vaccine_name) 
);


CREATE TABLE vaccine (
iso_code TEXT NOT NULL,
vaccine_name TEXT NOT NULL,
PRIMARY KEY (
vaccine_name,
iso_code
),
FOREIGN KEY (
iso_code
)
REFERENCES locations (iso_code) 
);
