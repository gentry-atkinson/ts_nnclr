# coding: utf-8

"""
    Lightly API

    lightly_plus_time.lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly_plus_time.lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly_plus_time.lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly_plus_time.lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from lightly_plus_time.lightly.openapi_generated.swagger_client.configuration import Configuration


class SharedAccessConfigData(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'id': 'MongoObjectID',
        'owner': 'str',
        'access_type': 'SharedAccessType',
        'users': 'list[str]',
        'organizations': 'list[str]',
        'created_at': 'Timestamp',
        'last_modified_at': 'Timestamp'
    }

    attribute_map = {
        'id': 'id',
        'owner': 'owner',
        'access_type': 'accessType',
        'users': 'users',
        'organizations': 'organizations',
        'created_at': 'createdAt',
        'last_modified_at': 'lastModifiedAt'
    }

    def __init__(self, id=None, owner=None, access_type=None, users=None, organizations=None, created_at=None, last_modified_at=None, _configuration=None):  # noqa: E501
        """SharedAccessConfigData - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._id = None
        self._owner = None
        self._access_type = None
        self._users = None
        self._organizations = None
        self._created_at = None
        self._last_modified_at = None
        self.discriminator = None

        self.id = id
        self.owner = owner
        self.access_type = access_type
        self.users = users
        self.organizations = organizations
        self.created_at = created_at
        self.last_modified_at = last_modified_at

    @property
    def id(self):
        """Gets the id of this SharedAccessConfigData.  # noqa: E501


        :return: The id of this SharedAccessConfigData.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this SharedAccessConfigData.


        :param id: The id of this SharedAccessConfigData.  # noqa: E501
        :type: MongoObjectID
        """
        if self._configuration.client_side_validation and id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def owner(self):
        """Gets the owner of this SharedAccessConfigData.  # noqa: E501

        Id of the user who owns the dataset  # noqa: E501

        :return: The owner of this SharedAccessConfigData.  # noqa: E501
        :rtype: str
        """
        return self._owner

    @owner.setter
    def owner(self, owner):
        """Sets the owner of this SharedAccessConfigData.

        Id of the user who owns the dataset  # noqa: E501

        :param owner: The owner of this SharedAccessConfigData.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and owner is None:
            raise ValueError("Invalid value for `owner`, must not be `None`")  # noqa: E501

        self._owner = owner

    @property
    def access_type(self):
        """Gets the access_type of this SharedAccessConfigData.  # noqa: E501


        :return: The access_type of this SharedAccessConfigData.  # noqa: E501
        :rtype: SharedAccessType
        """
        return self._access_type

    @access_type.setter
    def access_type(self, access_type):
        """Sets the access_type of this SharedAccessConfigData.


        :param access_type: The access_type of this SharedAccessConfigData.  # noqa: E501
        :type: SharedAccessType
        """
        if self._configuration.client_side_validation and access_type is None:
            raise ValueError("Invalid value for `access_type`, must not be `None`")  # noqa: E501

        self._access_type = access_type

    @property
    def users(self):
        """Gets the users of this SharedAccessConfigData.  # noqa: E501

        List of user mails with access to the dataset  # noqa: E501

        :return: The users of this SharedAccessConfigData.  # noqa: E501
        :rtype: list[str]
        """
        return self._users

    @users.setter
    def users(self, users):
        """Sets the users of this SharedAccessConfigData.

        List of user mails with access to the dataset  # noqa: E501

        :param users: The users of this SharedAccessConfigData.  # noqa: E501
        :type: list[str]
        """
        if self._configuration.client_side_validation and users is None:
            raise ValueError("Invalid value for `users`, must not be `None`")  # noqa: E501

        self._users = users

    @property
    def organizations(self):
        """Gets the organizations of this SharedAccessConfigData.  # noqa: E501

        List of organizations with access to the dataset  # noqa: E501

        :return: The organizations of this SharedAccessConfigData.  # noqa: E501
        :rtype: list[str]
        """
        return self._organizations

    @organizations.setter
    def organizations(self, organizations):
        """Sets the organizations of this SharedAccessConfigData.

        List of organizations with access to the dataset  # noqa: E501

        :param organizations: The organizations of this SharedAccessConfigData.  # noqa: E501
        :type: list[str]
        """
        if self._configuration.client_side_validation and organizations is None:
            raise ValueError("Invalid value for `organizations`, must not be `None`")  # noqa: E501

        self._organizations = organizations

    @property
    def created_at(self):
        """Gets the created_at of this SharedAccessConfigData.  # noqa: E501


        :return: The created_at of this SharedAccessConfigData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at):
        """Sets the created_at of this SharedAccessConfigData.


        :param created_at: The created_at of this SharedAccessConfigData.  # noqa: E501
        :type: Timestamp
        """
        if self._configuration.client_side_validation and created_at is None:
            raise ValueError("Invalid value for `created_at`, must not be `None`")  # noqa: E501

        self._created_at = created_at

    @property
    def last_modified_at(self):
        """Gets the last_modified_at of this SharedAccessConfigData.  # noqa: E501


        :return: The last_modified_at of this SharedAccessConfigData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._last_modified_at

    @last_modified_at.setter
    def last_modified_at(self, last_modified_at):
        """Sets the last_modified_at of this SharedAccessConfigData.


        :param last_modified_at: The last_modified_at of this SharedAccessConfigData.  # noqa: E501
        :type: Timestamp
        """
        if self._configuration.client_side_validation and last_modified_at is None:
            raise ValueError("Invalid value for `last_modified_at`, must not be `None`")  # noqa: E501

        self._last_modified_at = last_modified_at

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(SharedAccessConfigData, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, SharedAccessConfigData):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, SharedAccessConfigData):
            return True

        return self.to_dict() != other.to_dict()