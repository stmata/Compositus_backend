from utils.utils import (
    colorize)
from utils.db_service import MongoDBManager
from colorama import init as colorama_init, Fore, Style

mongo = MongoDBManager()
MASTER_COLL = mongo.get_collection("employees")
USERS_COLL  = mongo.get_collection("Admins")

def mark_all_users_update_start():
    """Met à jour tous les users: update_data=true, supprime update_error."""
    try:
        res = USERS_COLL.update_many({}, {"$set": {"update_data": True}, "$unset": {"update_error": ""}})
    except Exception as e:
        print(colorize(f"set update_data=true :: {e}", Fore.YELLOW, True))

def mark_all_users_update_done():
    try:
        res = USERS_COLL.update_many({}, {"$set": {"update_data": False}})
    except Exception as e:
        print(colorize(f"cannot set update_data=false :: {e}", Fore.YELLOW, True))

def mark_all_users_update_error(msg: str):
    try:
        res = USERS_COLL.update_many({}, {"$set": {"update_data": False, "update_error": msg}})
    except Exception as e:
        print(colorize(f"cannot set update_error :: {e}", Fore.YELLOW, True))
