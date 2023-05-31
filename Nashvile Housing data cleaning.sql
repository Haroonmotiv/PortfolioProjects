select* from NashvilleHousing

--Standardize the Date Format
select saledate2,CONVERT(Date,SaleDate)
from NashvilleHousing

UPDATE NashvilleHousing
SET SaleDate = CONVERT(Date,SaleDate)

ALTER table NashvilleHousing
add saledate2 date;

update NashvilleHousing
set saledate2 = CONVERT(Date,SaleDate)

--Populate Property Address 
select *
from NashvilleHousing
order by ParcelID

select a.ParcelID,a.PropertyAddress,b.ParcelID,b.PropertyAddress, ISNULL(a.PropertyAddress,b.PropertyAddress)
from NashvilleHousing a
JOIN NashvilleHousing b 
on a.ParcelID = b.ParcelID
AND a.[UniqueID ] <> b.[UniqueID ]
where a.PropertyAddress is null

update a
set PropertyAddress = ISNULL(a.PropertyAddress,b.PropertyAddress)
from NashvilleHousing a
JOIN NashvilleHousing b 
on a.ParcelID = b.ParcelID
AND a.[UniqueID ] <> b.[UniqueID ]

--Breaking out Address into individual columns (Address,City,State)
select PropertyAddress
from NashvilleHousing

--Where Property Address is NULL
--Order by ParcelID

SELECT
SUBSTRING(PropertyAddress,1,CHARINDEX(',',PropertyAddress)-1) as Address
,SUBSTRING(PropertyAddress,CHARINDEX(',',PropertyAddress)+1,len(PropertyAddress)) as Address
from NashvilleHousing

ALTER table NashvilleHousing
add PropertySplitAddress nvarchar(255);

update NashvilleHousing
set PropertySplitAddress = SUBSTRING(PropertyAddress,1,CHARINDEX(',',PropertyAddress)-1)

ALTER table NashvilleHousing
add PropertySplitCity nvarchar(255);

update NashvilleHousing
set PropertySplitCity = SUBSTRING(PropertyAddress,CHARINDEX(',',PropertyAddress)+1,len(PropertyAddress))

Select * from NashvilleHousing

Select OwnerAddress
from NashvilleHousing

Select 
PARSENAME(replace(OwnerAddress,',','.'),3) 
,PARSENAME(replace(OwnerAddress,',','.'),2)
,PARSENAME(replace(OwnerAddress,',','.'),1)
from NashvilleHousing


ALTER table NashvilleHousing
add OwnerSplitAddress nvarchar(255);

update NashvilleHousing
set OwnerSplitAddress = PARSENAME(replace(OwnerAddress,',','.'),3)

ALTER table NashvilleHousing
add OwnerSplitCity nvarchar(255);

update NashvilleHousing
set OwnerSplitCity = PARSENAME(replace(OwnerAddress,',','.'),2)

ALTER table NashvilleHousing
add OwnerSplitState nvarchar(255);

update NashvilleHousing
set OwnerSplitState = PARSENAME(replace(OwnerAddress,',','.'),1)

--Convert Y and N to Yes and No in column SoldAsVacant

select Distinct(SoldAsVacant),Count(SoldAsVacant)
from NashvilleHousing
group by SoldAsVacant
order by 2

select SoldAsVacant,
CASE 
    when SoldAsVacant = 'Y' THEN 'Yes'
	when SoldAsVacant = 'N' THEN 'No'
	ELSE 
	SoldAsVacant
	END
from NashvilleHousing

UPDATE NashvilleHousing
SET SoldAsVacant = CASE 
    when SoldAsVacant = 'Y' THEN 'Yes'
	when SoldAsVacant = 'N' THEN 'No'
	ELSE 
	SoldAsVacant
	END

--Remove Duplicates

WITH RowNumCTE AS (
select *,
       ROW_NUMBER() OVER(
	   PARTITION BY ParcelID,
	                PropertyAddress,
					SalePrice,
					SaleDate,
					LegalReference
					ORDER BY UniqueID) row_num
from NashvilleHousing
--Order By ParcelID
)
Select *
From RowNumCTE
Where row_num>1
Order by PropertyAddress

--Delete Unused Columns

ALTER table NashvilleHousing
DROP COLUMN SaleDate,OwnerAddress,TaxDistrict,PropertyAddress
