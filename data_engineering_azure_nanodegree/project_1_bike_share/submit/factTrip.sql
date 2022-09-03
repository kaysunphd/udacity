IF OBJECT_ID('dbo.factTrip') IS NOT NULL
BEGIN
    DROP TABLE factTrip
END
GO;

CREATE TABLE factTrip
(
    [date_key] INT,
    [rider_key] BIGINT,
    [start_station_key] NVARCHAR(4000),
    [end_station_key] NVARCHAR(4000),
    [rider_age] SMALLINT,
    [trip_duration] INT
)
GO;

INSERT INTO factTrip (date_key, rider_key, start_station_key, end_station_key, rider_age, trip_duration)
SELECT d.date_id                                                                                                       AS date_key,
       r.rider_id                                                                                                      AS rider_key,
       t.start_station_id                                                                                              AS start_station_key,
       t.end_station_id                                                                                                AS end_station_key,
       DATEDIFF(YEAR, TRY_CONVERT(DATE, r.birthday), TRY_CONVERT(DATE, t.started_at))                                  AS rider_age,
       DATEDIFF(MINUTE, TRY_CONVERT(DATETIME2, LEFT(t.started_at, 18)), TRY_CONVERT(DATETIME2, LEFT(t.ended_at, 18)))  AS trip_duration
FROM staging_trip t
JOIN staging_rider r  ON (r.rider_id = t.rider_id)
JOIN staging_payment p  ON (p.rider_id = t.rider_id)
JOIN dimDate d ON (d.Date = p.date)
GO;