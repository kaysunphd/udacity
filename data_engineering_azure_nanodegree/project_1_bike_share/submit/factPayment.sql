IF OBJECT_ID('dbo.factPayment') IS NOT NULL
BEGIN
    DROP TABLE factPayment
END
GO;

CREATE TABLE factPayment
(
    [date_key] INT,
    [rider_key] BIGINT,
    [amount] FLOAT
)
GO;

INSERT INTO factPayment (date_key, rider_key, amount)
SELECT d.date_id               AS date_key,
       r.rider_id           AS rider_key,
       p.amount             AS amount
FROM staging_payment p
JOIN staging_rider r  ON (p.rider_id = r.rider_id)
JOIN dimDate d ON (d.Date = p.date)
GO;