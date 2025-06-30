from firebase_admin import firestore
import hashlib

db = firestore.client()

def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def is_password_valid(password):
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not any(c.isdigit() for c in password):
        return False, "Password must include at least one number."
    if not any(c.isalpha() for c in password):
        return False, "Password must include at least one letter."
    return True, ""


def register_user(username, password):
    is_valid, msg = is_password_valid(password)
    if not is_valid:
        return False, msg

    user_ref = db.collection("users").document(username)
    if user_ref.get().exists:
        return False, "Username already exists."

    user_ref.set({"password": hash_password(password)})
    return True, ""


def validate_user(username, password):
    """Validates login credentials."""
    doc = db.collection("users").document(username).get()
    if not doc.exists:
        return False
    return doc.to_dict().get("password") == hash_password(password)


def change_password(username, old_password, new_password):
    if not validate_user(username, old_password):
        return False, "Current password is incorrect."

    is_valid, msg = is_password_valid(new_password)
    if not is_valid:
        return False, msg

    user_ref = db.collection("users").document(username)
    user_ref.update({"password": hash_password(new_password)})
    return True, "Password changed successfully."
