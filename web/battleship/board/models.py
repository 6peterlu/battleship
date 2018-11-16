from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.

class Board(models.Model):
	board_data = ArrayField(
        ArrayField(
            models.IntegerField(),
            size=10,
        ),
        size=10,
    )

class Game(models.Model):
	user_board = Board()
	CPU_board = Board()
	is_complete = models.BooleanField()
	winner = models.IntegerField()
	num_moves = models.IntegerField()

class User(models.Model):
	user_id = models.CharField(max_length=100)
	user_game = Game()

class UserName(models.Model):
	user_name = models.CharField(max_length=100)
