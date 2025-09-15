--D.1)
SELECT DISTINCT
l.location_name AS "Country Name (CN)", 
c.total_boosters AS "Total Vaccinations (administered to date)", 
c.people_vaccinated AS "Daily Vaccinations",
c.date AS "Date"
FROM 
country_vaccinations_total c
JOIN 
locations l ON c.iso_code = l.iso_code
WHERE 
c.people_vaccinated > (SELECT AVG(people_vaccinated) FROM country_vaccinations_total)
ORDER BY 
"Daily Vaccinations" DESC;


--D.2)
SELECT l.location_name AS Country, SUM(v.totalVaccinations) AS Cumulative_Doses
FROM vaccination v
JOIN locations l ON v.iso_code = l.iso_code
GROUP BY l.location_name
HAVING SUM(v.totalVaccinations) > (
SELECT AVG(totalVaccinations) FROM vaccination
)
ORDER BY Cumulative_Doses DESC;


--D.3)
SELECT 
l.location_name AS "Country", 
c.vaccine_name AS "Vaccine Type"
FROM 
country_vaccinations_total c
JOIN 
locations l ON c.iso_code = l.iso_code
GROUP BY 
l.location_name, c.vaccine_name
ORDER BY 
"Country", "Vaccine Type";


--D.4)
SELECT l.location_name AS Country, 
cvt.source_url AS "Source Name (URL)", 
MAX(CAST(cvt.people_vaccinated AS INTEGER)) AS "Biggest total Administered Vaccines"
FROM country_vaccinations_total cvt
JOIN locations l ON cvt.iso_code = l.iso_code
GROUP BY cvt.iso_code, cvt.source_url
ORDER BY cvt.source_url, l.location_name;


--D.5)
SELECT 
	strftime('%Y-%W', v.date) AS "Date Range (Weeks)", 
	SUM(CASE WHEN l.location_name = 'Australia' THEN v.people_fully_vaccinated ELSE 0 END) AS Australia,
	SUM(CASE WHEN l.location_name = 'Germany' THEN v.people_fully_vaccinated ELSE 0 END) AS Germany,
	SUM(CASE WHEN l.location_name = 'United Kingdom' THEN v.people_fully_vaccinated ELSE 0 END) AS England, 
	SUM(CASE WHEN l.location_name = 'France' THEN v.people_fully_vaccinated ELSE 0 END) AS France
FROM vaccination v
JOIN locations l ON v.iso_code = l.iso_code
WHERE strftime('%Y', v.date) IN ('2021', '2022') 
AND l.location_name IN ('Australia', 'Germany', 'United Kingdom', 'France')
GROUP BY strftime('%Y-%W', v.date)
ORDER BY strftime('%Y-%W', v.date);


