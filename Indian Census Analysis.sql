select * from dbo.Data1
select * from dbo.Data2

--Number of rows into our dataset
select count(*) ColumnCount1 from PortfolioProject..data1 
select count(*) ColumnCount2 from PortfolioProject..data2 

--Dataset in Jharkhand and Bihar
select * from PortfolioProject..data1 where state in ('jharkhand','bihar')

--Population of India
select sum(population) TotalPopulation from PortfolioProject..data2

--Average Growth of India
select state,avg(Growth)*100 AvgGrowth from PortfolioProject..data1 group by state

--Average Sex Ratio of India
select state,round(avg(Sex_Ratio),0) AvgSexRatio from PortfolioProject..data1 group by state order by AvgSexRatio desc

--Average Literacy Rate of India
select state,round(avg(Literacy),0) AvgLiteracy from PortfolioProject..data1 group by state order by AvgLiteracy desc
select state,round(avg(Literacy),0) AvgLiteracy from PortfolioProject..data1 group by state having round(avg(Literacy),0)>90 order by AvgLiteracy desc

--Top 3 states showing highest growth ratio
select top 3 state,avg(Growth)*100 AvgGrowth from PortfolioProject..data1 group by state order by AvgGrowth desc

--Bottom 3 states showing lowest sex ratio
select top 3 state,round(avg(Sex_Ratio),0) AvgSexRatio from PortfolioProject..data1 group by state order by AvgSexRatio asc

--Top and Bottom 3 states in literacy rate
select top 3 state,round(avg(Literacy),0) AvgLiteracy from PortfolioProject..data1 group by state order by AvgLiteracy desc
select top 3 state,round(avg(Literacy),0) AvgLiteracy from PortfolioProject..data1 group by state order by AvgLiteracy asc

drop table if exists #topstates
create table #topstates
( state nvarchar(255),
topstate float)

insert into #topstates
select state,round(avg(Literacy),0) AvgLiteracy from PortfolioProject..data1 group by state order by AvgLiteracy desc
select top 3 * from #topstates order by #topstates.topstate desc

-----------------------------------------------------
drop table if exists #bottomstates
create table #bottomstates
( state nvarchar(255),
bottomstate float)

insert into #bottomstates
select state,round(avg(Literacy),0) AvgLiteracy from PortfolioProject..data1 group by state order by AvgLiteracy desc
select top 3 * from #bottomstates order by #bottomstates.bottomstate asc

--Union
select * from(
select top 3 * from #topstates order by #topstates.topstate desc) a
union
select * from(
select top 3 * from #bottomstates order by #bottomstates.bottomstate asc)b

--States starting with letter a or b
select distinct state from PortfolioProject..data1 where lower(state) like 'a%' or lower(state) like 'b%'

--States ending with letter d
select distinct state from PortfolioProject..data1 where lower(state) like '%d'

--States starting with letter a and ends with letter m
select distinct state from PortfolioProject..data1 where lower(state) like 'a%' and lower(state) like '%m'

--Joining both tables
--Total Males and Females
select d.State,sum(d.males) Total_males, sum(d.females) Total_females from
(select c.District,c.State,round(c.Population/(c.sex_ratio+1),0) males,round((c.population*c.Sex_Ratio)/(c.Sex_Ratio+1),0) females from
(select a.District,a.State,a.Sex_Ratio/1000 sex_ratio ,b.Population from PortfolioProject..data1 a 
inner join PortfolioProject..data2 b on a.District = b.District ) c)d
group by d.State

--Total Literacy rate
select d.State,sum(Literate_People) Total_Literate,sum(Illiterate_People) Total_Illiterate from
(select c.District,c.state, round(c.Literacy_Ratio*c.population,0) Literate_People, round((1-c.Literacy_Ratio)*c.Population,0) Illiterate_People from
(select a.District,a.State,a.Literacy/100 Literacy_Ratio,b.Population from data1 a inner join data2 b on a.District = b.District) c) d
group by d.State

--Population in previous census
select sum(m.previous_census_population) Total_pop_prev,sum(m.current_population) Total_pop_cur from
(select e.state,sum(e.previous_census_population) previous_census_population, sum(e.current_population) current_population from
(select d.district, d.state, round(d.population/(1+d.growth),0) previous_census_population, d.population current_population from 
(select a.District,a.State,a.Growth growth ,b.Population from PortfolioProject..data1 a 
inner join PortfolioProject..data2 b on a.District = b.District)d)e
group by e.State)m

--Population Vs Area
select (g.total_area/g.Total_pop_prev) as previous_census_population_vs_area,(g.total_area/g.Total_pop_cur) as current_census_population_vs_area from
(select q.*,r.total_area from(
(select '1' as keyy,n.* from
(select sum(m.previous_census_population) Total_pop_prev,sum(m.current_population) Total_pop_cur from
(select e.state,sum(e.previous_census_population) previous_census_population, sum(e.current_population) current_population from
(select d.district, d.state, round(d.population/(1+d.growth),0) previous_census_population, d.population current_population from 
(select a.District,a.State,a.Growth growth ,b.Population from PortfolioProject..data1 a 
inner join PortfolioProject..data2 b on a.District = b.District)d)e
group by e.State)m)n)q inner join ( 

select '1' as keyy, z.* from (
select sum(area_km2) total_area from data2)z) r on q.keyy = r.keyy))g

--Window
--output top 3 districts from each state with highest literacy rate

select a.* from
(select district,state,literacy,rank() over(partition by state order by literacy desc) rnk from data1)a
where a.rnk in (1,2,3) order by state
