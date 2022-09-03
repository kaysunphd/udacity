IF OBJECT_ID('dbo.dimDate') IS NOT NULL
BEGIN
    DROP TABLE dimDate
END
GO;

/*derivative of
https://datasavvy.me/2016/08/06/create-a-date-dimension-in-azure-sql-data-warehouse
to create date dimension table in Azure Synapse SQL Pool.
*/

DECLARE @StartDate DATE = '2013-01-01', @NumberOfYears INT = 30

CREATE TABLE #dimDateTemp
(
  [date]       DATE,  
  [day]        TINYINT,
  [month]      TINYINT,
  FirstOfMonth DATE,
  [MonthName]  VARCHAR(12),
  [week]       TINYINT,
  [ISOweek]    TINYINT,
  [DayOfWeek]  TINYINT,
  [quarter]    TINYINT,
  [year]       SMALLINT,
  FirstOfYear  DATE,
  Style112     char(8),
  Style101     char(10)
)

SET DATEFIRST 7;
SET DATEFORMAT mdy;
SET LANGUAGE US_ENGLISH;

DECLARE @CutoffDate DATE = DATEADD(YEAR, @NumberOfYears, @StartDate);

INSERT #dimDateTemp([date]) 
SELECT d
FROM
(
  SELECT d = DATEADD(DAY, rn-1, @StartDate)
  FROM 
  (
    SELECT TOP (DATEDIFF(DAY, @StartDate, @CutoffDate)) 
      rn = ROW_NUMBER() OVER (ORDER BY s1.[object_id])
    FROM sys.all_objects AS s1
    CROSS JOIN sys.all_objects AS s2
    ORDER BY s1.[object_id]
  ) AS x
) AS y

UPDATE #dimDateTemp 
set 
  [day]        = DATEPART(DAY,      [date]),
  [month]      = DATEPART(MONTH,    [date]),
  FirstOfMonth = CONVERT(DATE, DATEADD(MONTH, DATEDIFF(MONTH, 0, [date]), 0)),
  [MonthName]  = DATENAME(MONTH,    [date]),
  [week]       = DATEPART(WEEK,     [date]),
  [ISOweek]    = DATEPART(ISO_WEEK, [date]),
  [DayOfWeek]  = DATEPART(WEEKDAY,  [date]),
  [quarter]    = DATEPART(QUARTER,  [date]),
  [year]       = DATEPART(YEAR,     [date]),
  FirstOfYear  = CONVERT(DATE, DATEADD(YEAR,  DATEDIFF(YEAR,  0, [date]), 0)),
  Style112     = CONVERT(CHAR(8),   [date], 112),
  Style101     = CONVERT(CHAR(10),  [date], 101)

CREATE TABLE dimDate
WITH
(
    DISTRIBUTION = ROUND_ROBIN
)
AS
SELECT
  date_id       = CONVERT(INT, Style112),
  [Date]        = [date],
  [Day]         = CONVERT(TINYINT, [day]),
  [Weekday]     = CONVERT(TINYINT, [DayOfWeek]),
  [WeekDayName] = CONVERT(VARCHAR(10), DATENAME(WEEKDAY, [date])),
  [DOWInMonth]  = CONVERT(TINYINT, ROW_NUMBER() OVER 
                  (PARTITION BY FirstOfMonth, [DayOfWeek] ORDER BY [date])),
  [DayOfYear]   = CONVERT(SMALLINT, DATEPART(DAYOFYEAR, [date])),
  WeekOfMonth   = CONVERT(TINYINT, DENSE_RANK() OVER 
                  (PARTITION BY [year], [month] ORDER BY [week])),
  WeekOfYear    = CONVERT(TINYINT, [week]),
  ISOWeekOfYear = CONVERT(TINYINT, ISOWeek),
  [Month]       = CONVERT(TINYINT, [month]),
  [MonthName]   = CONVERT(VARCHAR(10), [MonthName]),
  [Quarter]     = CONVERT(TINYINT, [quarter]),
  QuarterName   = CONVERT(VARCHAR(6), CASE [quarter] WHEN 1 THEN 'First' 
                  WHEN 2 THEN 'Second' WHEN 3 THEN 'Third' WHEN 4 THEN 'Fourth' END), 
  [Year]        = [year],
  MMYYYY        = CONVERT(CHAR(6), LEFT(Style101, 2)    + LEFT(Style112, 4)),
  MonthYear     = CONVERT(CHAR(8), LEFT([MonthName], 3) + ' ' + LEFT(Style112, 4))

FROM #dimDateTemp

DROP Table #dimDateTemp

GO;