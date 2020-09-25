-- BDA 696 Assignment 2

-- Use baseball database to create 3 tables

use baseball;

-- First table: Career Batting Average

drop table if exists CareerBA;

create table CareerBA
select batter, sum(Hit)/sum(atBat) as cBA
from batter_counts where atBat>0
group by batter;

select * from CareerBA;

-- Second table: Seasonal Batting Average

drop table if exists SeasonalBA;

create table SeasonalBA
select batter_counts.batter, sum(batter_counts.Hit)/sum(batter_counts.atBat) as sBA, 
year(game.local_date) as Season
from batter_counts join game on batter_counts.game_id=game.game_id where atBat>0
group by Season, batter
order by batter, Season;

select * from SeasonalBA;

-- Third table: Rolling Batting Average

drop table if exists RollingBA;

create table RollingBA
select batter, date(game.local_date) as gameday,
sum(batter_counts.Hit) over (partition by batter
						     	order by date(game.local_date)
							    range between interval '101' day preceding
						    	and interval '1' day preceding)
						/
sum(batter_counts.atBat) over (partition by batter
							    order by date(game.local_date)
							    range between interval '101' day preceding
							    and interval '1' day preceding)
as rBA
from batter_counts join game on batter_counts.game_id = game.game_id where atBat > 0
group by gameday, batter
order by gameday, batter;

select * from RollingBA;




	
	
	



