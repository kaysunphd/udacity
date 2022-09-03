IF OBJECT_ID('dbo.dimStation') IS NOT NULL
BEGIN
    DROP TABLE dimStation
END
GO;

CREATE TABLE dimStation
(
    [station_id] NVARCHAR(4000),
    [station_name] NVARCHAR(4000),
    [latitude] FLOAT,
    [longitude] FLOAT
)
GO;

INSERT INTO dimStation (station_id, station_name, latitude, longitude)
SELECT [station_id],
       [name]             AS station_name,
       [latitude],
       [longitude]
FROM staging_station
GO;