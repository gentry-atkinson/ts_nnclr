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


class Embedding2dCreateRequest(object):
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
        'name': 'str',
        'dimensionality_reduction_method': 'DimensionalityReductionMethod',
        'coordinates_dimension1': 'Embedding2dCoordinates',
        'coordinates_dimension2': 'Embedding2dCoordinates'
    }

    attribute_map = {
        'name': 'name',
        'dimensionality_reduction_method': 'dimensionalityReductionMethod',
        'coordinates_dimension1': 'coordinatesDimension1',
        'coordinates_dimension2': 'coordinatesDimension2'
    }

    def __init__(self, name=None, dimensionality_reduction_method=None, coordinates_dimension1=None, coordinates_dimension2=None, _configuration=None):  # noqa: E501
        """Embedding2dCreateRequest - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._name = None
        self._dimensionality_reduction_method = None
        self._coordinates_dimension1 = None
        self._coordinates_dimension2 = None
        self.discriminator = None

        self.name = name
        self.dimensionality_reduction_method = dimensionality_reduction_method
        self.coordinates_dimension1 = coordinates_dimension1
        self.coordinates_dimension2 = coordinates_dimension2

    @property
    def name(self):
        """Gets the name of this Embedding2dCreateRequest.  # noqa: E501

        Name of the 2d embedding (default is embedding name + __2d)  # noqa: E501

        :return: The name of this Embedding2dCreateRequest.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this Embedding2dCreateRequest.

        Name of the 2d embedding (default is embedding name + __2d)  # noqa: E501

        :param name: The name of this Embedding2dCreateRequest.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def dimensionality_reduction_method(self):
        """Gets the dimensionality_reduction_method of this Embedding2dCreateRequest.  # noqa: E501


        :return: The dimensionality_reduction_method of this Embedding2dCreateRequest.  # noqa: E501
        :rtype: DimensionalityReductionMethod
        """
        return self._dimensionality_reduction_method

    @dimensionality_reduction_method.setter
    def dimensionality_reduction_method(self, dimensionality_reduction_method):
        """Sets the dimensionality_reduction_method of this Embedding2dCreateRequest.


        :param dimensionality_reduction_method: The dimensionality_reduction_method of this Embedding2dCreateRequest.  # noqa: E501
        :type: DimensionalityReductionMethod
        """
        if self._configuration.client_side_validation and dimensionality_reduction_method is None:
            raise ValueError("Invalid value for `dimensionality_reduction_method`, must not be `None`")  # noqa: E501

        self._dimensionality_reduction_method = dimensionality_reduction_method

    @property
    def coordinates_dimension1(self):
        """Gets the coordinates_dimension1 of this Embedding2dCreateRequest.  # noqa: E501


        :return: The coordinates_dimension1 of this Embedding2dCreateRequest.  # noqa: E501
        :rtype: Embedding2dCoordinates
        """
        return self._coordinates_dimension1

    @coordinates_dimension1.setter
    def coordinates_dimension1(self, coordinates_dimension1):
        """Sets the coordinates_dimension1 of this Embedding2dCreateRequest.


        :param coordinates_dimension1: The coordinates_dimension1 of this Embedding2dCreateRequest.  # noqa: E501
        :type: Embedding2dCoordinates
        """
        if self._configuration.client_side_validation and coordinates_dimension1 is None:
            raise ValueError("Invalid value for `coordinates_dimension1`, must not be `None`")  # noqa: E501

        self._coordinates_dimension1 = coordinates_dimension1

    @property
    def coordinates_dimension2(self):
        """Gets the coordinates_dimension2 of this Embedding2dCreateRequest.  # noqa: E501


        :return: The coordinates_dimension2 of this Embedding2dCreateRequest.  # noqa: E501
        :rtype: Embedding2dCoordinates
        """
        return self._coordinates_dimension2

    @coordinates_dimension2.setter
    def coordinates_dimension2(self, coordinates_dimension2):
        """Sets the coordinates_dimension2 of this Embedding2dCreateRequest.


        :param coordinates_dimension2: The coordinates_dimension2 of this Embedding2dCreateRequest.  # noqa: E501
        :type: Embedding2dCoordinates
        """
        if self._configuration.client_side_validation and coordinates_dimension2 is None:
            raise ValueError("Invalid value for `coordinates_dimension2`, must not be `None`")  # noqa: E501

        self._coordinates_dimension2 = coordinates_dimension2

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
        if issubclass(Embedding2dCreateRequest, dict):
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
        if not isinstance(other, Embedding2dCreateRequest):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, Embedding2dCreateRequest):
            return True

        return self.to_dict() != other.to_dict()
