IF OBJECT_ID('dbo.dimRider') IS NOT NULL
BEGIN
    DROP TABLE dimRider
END
GO;

CREATE TABLE dimRider
(
    [rider_id] BIGINT,
    [first_name] NVARCHAR(4000),
    [last_name] NVARCHAR(4000),
    [address] NVARCHAR(50),
    [birthday] DATE,
    [account_start_date] DATE,
    [account_end_date] DATE,
    [is_member] BIT
)
GO;

INSERT INTO dimRider (rider_id, first_name, last_name, address, birthday, account_start_date, account_end_date, is_member)
SELECT [rider_id],
       [first_name],
       [last_name],
       [address],
       TRY_CONVERT(DATE, LEFT(birthday, 10))  AS birthday,
       TRY_CONVERT(DATE, LEFT(account_start_date, 10))  AS account_start_date,
       TRY_CONVERT(DATE, LEFT(account_end_date, 10))  AS account_end_date,
       [is_member]
FROM staging_rider
GO;